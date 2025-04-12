import logging
import os
import re
import json
import time
import copy
import psycopg2
from psycopg2 import sql
import pandas as pd
import numpy as np
import ast
from rag import settings
from rag.settings import PAGERANK_FLD
from rag.utils import singleton
from api.utils.file_utils import get_project_base_directory
import traceback
from api.utils import get_base_config  

from rag.utils.doc_store_conn import (
    DocStoreConnection,
    MatchExpr,
    MatchTextExpr,
    MatchDenseExpr,
    FusionExpr,
    OrderByExpr,
)

logger = logging.getLogger('ragflow.opengauss_conn')

ATTEMPT_TIME = 2

def equivalent_condition_to_str(condition: dict, table_columns: dict = None) -> str:
    assert "_id" not in condition

    def exists(column):
        assert column in table_columns, f"'{column}' should be in '{table_columns}'."
        column_type, default_value = table_columns[column]
        if "char" in column_type.lower():  
            if not default_value:
                default_value = ""
            return f"{column} != '{default_value}'"
        return f"{column} != {default_value}"

    conditions = []
    for key, value in condition.items():
        if not isinstance(key, str) or key == "kb_id" or (value is None or value == ""):
            continue

        if isinstance(value, list):
            in_conditions = [f"'{item}'" if isinstance(item, str) else str(item) for item in value]
            if in_conditions:
                str_in_conditions = ", ".join(in_conditions)
                conditions.append(f"{key} IN ({str_in_conditions})")
        elif key == "must_not" and isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_key == "exists":
                    conditions.append(f"NOT ({exists(sub_value)})")
        elif isinstance(value, str):
            conditions.append(f"{key} = '{value}'")
        elif key == "exists":
            conditions.append(exists(value))
        else:
            conditions.append(f"{key} = {value}")

    return " AND ".join(conditions) if conditions else "1=1"


def concat_dataframes(df_list: list[pd.DataFrame], selectFields: list[str]) -> pd.DataFrame:
    df_list2 = [df for df in df_list if not df.empty]
    if df_list2:
        return pd.concat(df_list2, axis=0).reset_index(drop=True)

    schema = []
    for field_name in selectFields:
        if field_name == 'score()':  # Workaround: fix schema is changed to score()
            schema.append('SCORE')
        elif field_name == 'similarity()':  # Workaround: fix schema is changed to similarity()
            schema.append('SIMILARITY')
        else:
            schema.append(field_name)
    return pd.DataFrame(columns=schema)

def get_tsquery(query: str)-> str:
    clean_text = re.sub(r'\^[\d.]+|~[\d.]', '', query) 
    pattern = re.compile(r'[A-Za-z]+|[\u4e00-\u9fff]+|\d+')
    tokens = pattern.findall(clean_text)
    keywords = [token for token in tokens if token.upper() != 'OR']
    seen = set()
    unique_keywords = []
    for token in keywords:
        lower = token.lower()
        if lower not in seen:
            seen.add(lower)
            unique_keywords.append(token)
    result = " ".join(unique_keywords)
    return result

def generate_textsearch_query(weight_fields, search_term, output, knowledgebaseId, table_name, limit, text_weight, filter_cond):
    field_weights = []
    for item in weight_fields:
        if '^' in item:
            field, weight = item.split('^')
            field_weights.append((field, float(weight)))
        else:
            field_weights.append((item, 1.0))
    
    subqueries = []
    for i, (field, weight) in enumerate(field_weights, start=1):
        subquery = f"""
        (
            SELECT /*+ indexscan({table_name} text_{knowledgebaseId}_{field})*/ {output}, ({field} <&> '{search_term}') * {weight} AS field_score 
            FROM {table_name} 
            WHERE {filter_cond}
            ORDER BY {field} <&> '{search_term}' DESC 
            LIMIT 200
        )"""
        subqueries.append(subquery)
    
    union_query = " UNION ".join(subqueries)
    
    full_query = f"""
        WITH combined_text AS 
        (
            {union_query}
        ),
        max_score AS (
            SELECT MAX(field_score) AS max_value
            FROM combined_text
        ),
        id_scores AS (
        SELECT 
            id, 
            MAX(field_score) AS max_raw_score  -- 计算所有id中的最大总分
        FROM combined_text
        GROUP BY id
        ),
        normalized_scores AS (
            SELECT 
                c.*,
                (i.max_raw_score / m.max_value) AS score  -- 归一化分数
            FROM combined_text c
            JOIN id_scores i ON c.id = i.id
            CROSS JOIN max_score m
        ),
        ranked_results AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY id ORDER BY score DESC) AS rn  -- 每个id内按原始score排序
            FROM normalized_scores
        )
        SELECT {output}, score
        FROM ranked_results
        WHERE rn = 1
        ORDER BY (score) DESC
        LIMIT {limit}
    """
    return full_query

@singleton
class OpenGaussConnection(DocStoreConnection):
    def __init__(self):
        self.info = {}
        logger.info(f"Use openGauss {settings.OG['host']} as the doc engine.")
        for _ in range(ATTEMPT_TIME):
            try:
                self.conn = psycopg2.connect(
                    host=settings.OG["host"],
                    port=settings.OG["port"],
                    user=settings.OG["user"],
                    password=settings.OG["password"],
                    dbname=settings.OG["database"]
                )
                if self.conn:
                    self.info = self.conn.info
                    break
            except Exception as e:
                logger.warning(f"{str(e)}. Waiting openGauss to be healthy.")
                time.sleep(5)
        if not self.conn:
            msg = f"openGauss is unhealthy in 120s."
            logger.error(msg)
            raise Exception(msg)
        self.enable_pq = settings.OG["enablepq"]
        self.conn.autocommit = True

        logger.info(f"openGauss is healthy.")

    """
    Database operations
    """

    def dbType(self) -> str:
        return "opengauss"

    def health(self) -> dict:
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return {"type": "opengauss", "status": "healthy" if result else "unhealthy"}
        except Exception as e:
            logger.error(f"openGauss health check failed: {str(e)}")
            return {"type": "opengauss", "status": "unhealthy"}
    """
    Table operations
    """

    def createIdx(self, indexName: str, knowledgebaseId: str, vectorSize: int):
        self.check_reconnect()

        table_name = f"{indexName}_{knowledgebaseId}"
        vector_name = f"q_{vectorSize}_vec"

        columns = []
        indices = []

        fp_mapping = os.path.join(
            get_project_base_directory(), "conf", "infinity_mapping.json"
        )
        if not os.path.exists(fp_mapping):
            raise Exception(f"Mapping file not found at {fp_mapping}")
        schema = json.load(open(fp_mapping))
        schema[vector_name] = {"type": f"vector({vectorSize})"}

        for field_name, field_info in schema.items():
            default_value = field_info.get("default")
            if default_value is not None:
                if field_info.get("type").lower() == "varchar":
                    field_info["type"] = "text"
                    default_clause = f"DEFAULT '{default_value}'"
                else:
                    default_clause = f"DEFAULT {default_value}"
            else:
                default_clause = ""

            column_definition = f"{field_name} {field_info['type']} {default_clause}"
            columns.append(column_definition)

            if field_info.get("analyzer"):
                indices.append({
                    "field": field_name,
                    "analyzer": field_info["analyzer"]
                })
        create_table_sql = sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {} (
                        {}
                    );
                """).format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(sql.SQL(column) for column in columns))
        with self.conn.cursor() as cursor:
            # create table
            cursor.execute(create_table_sql)

            # create fulltext index
            weight_map = ['title_tks', 'title_sm_tks', 'important_kwd', 'important_tks', 'question_tks', 'content_ltks', 'content_sm_ltks', 'content_with_weight']
            sql_parts = []
            for field in weight_map:
                cursor.execute(sql.SQL("""
                                        ALTER TABLE {} SET (parallel_workers = 32);
                                        CREATE INDEX IF NOT EXISTS {} ON {} USING bm25(({}));    
                                    """).format(
                    sql.Identifier(table_name),
                    sql.Identifier(f"text_{knowledgebaseId}_{field}"),
                    sql.Identifier(table_name),
                    sql.Identifier(field)
                ))
            self.conn.commit()
        logger.info(
            f"openGauss created table {table_name}, vector size {vectorSize}"
        )

    def deleteIdx(self, indexName: str, knowledgebaseId: str):
        table_name = f"{indexName}_{knowledgebaseId}"
        with self.conn.cursor() as cursor:
            cursor.execute(sql.SQL("""
                DROP TABLE IF EXISTS {}
            """).format(
                sql.Identifier(table_name)
            ))
            self.conn.commit()
        logger.info(f"openGauss dropped table {table_name}")

    def indexExist(self, indexName: str, knowledgebaseId: str) -> bool:
        table_name = f"{indexName}_{knowledgebaseId[:-10]}%"
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(sql.SQL("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_name like %s
                    )
                """), (table_name,))
                exists = cursor.fetchone()[0]
                return exists
            except Exception as e:
                print(f"openGauss indexExist error: {str(e)}")
                return False

    """
    CRUD operations
    """    
    def search(
            self, selectFields: list[str],
            highlightFields: list[str],
            condition: dict,
            matchExprs: list[MatchExpr],
            orderBy: OrderByExpr,
            offset: int,
            limit: int,
            indexNames: str | list[str],
            knowledgebaseIds: list[str],
            aggFields: list[str] = [],
            rank_feature: dict | None = None) -> tuple[pd.DataFrame, int]:
        self.check_reconnect()

        if isinstance(indexNames, str):
            indexNames = indexNames.split(",")
        assert isinstance(indexNames, list) and len(indexNames) > 0
        logger.info(f"[offset, limit]: {offset} {limit}")

        df_list = list()
        table_list = list()
        output = selectFields.copy()
        for essential_field in ["id"]:
            if essential_field not in output:
                output.append(essential_field)

        if matchExprs:
            if PAGERANK_FLD not in output:
                output.append(PAGERANK_FLD)
            output = [f for f in output if f != "_score"]

        # prepare filter conditions
        filter_cond = None
        filter_fulltext = ""
        if condition:
                table_ref = f"{indexNames[0]}_{knowledgebaseIds[0]}"
                table_columns = self.get_columns(table_ref)
                filter_cond = equivalent_condition_to_str(condition, table_columns)
    
        vector_similarity_weight = 0.5
        for matchExpr in matchExprs:
            if isinstance(matchExpr, MatchTextExpr):
                minimum_should_match = matchExpr.extra_options.get("minimum_should_match", 0.0)
                if isinstance(minimum_should_match, float):
                    str_minimum_should_match = str(int(minimum_should_match * 100)) + "%"
                    matchExpr.extra_options["minimum_should_match"] = str_minimum_should_match
                for k, v in matchExpr.extra_options.items():
                    if not isinstance(v, str):
                        matchExpr.extra_options[k] = str(v)
                logger.debug(f"openGauss search MatchTextExpr: {json.dumps(matchExpr.__dict__)}")
            elif isinstance(matchExpr, MatchDenseExpr):
                for k, v in matchExpr.extra_options.items():
                    if not isinstance(v, str):
                        matchExpr.extra_options[k] = str(v)
                similarity = matchExpr.extra_options.get("similarity")
                if similarity:
                    matchExpr.extra_options["threshold"] = similarity
                    del matchExpr.extra_options["similarity"]
                logger.debug(f"openGauss search MatchDenseExpr: {json.dumps(matchExpr.__dict__)}")
            elif isinstance(matchExpr, FusionExpr) and matchExpr.method == "weighted_sum" and "weights" in matchExpr.fusion_params:
                assert len(matchExprs) == 3 and isinstance(matchExprs[0], MatchTextExpr) and isinstance(matchExprs[1],
                                                                                                        MatchDenseExpr) and isinstance(
                    matchExprs[2], FusionExpr)
                weights = matchExpr.fusion_params["weights"]
                vector_similarity_weight = float(weights.split(",")[1])
                logging.info(f"vector_similarity_weight: {vector_similarity_weight}")
                logging.info(f"openGauss search FusionExpr: {json.dumps(matchExpr.__dict__)}")

        # construct a basic query template

        base_query = """ 
        set enable_seqscan=off;
        set bm25_topk = 200; 
        WITH combined AS (
            ({fulltext_sql}
        ) UNION ALL(
            SELECT {fields}, (1 - {vector_score}) AS score
            FROM {table_name}
            WHERE {condition} AND {vector_condition}
            ORDER BY {vector_score}   
            LIMIT {vector_topn})
        )
        SELECT * FROM combined
        ORDER BY score DESC
        OFFSET %s LIMIT %s;
        """ 

        total_hits_count = 0
        vector_column_name = ''
        text_score_expr = ""
        vector_score_expr = ""       

        for indexName in indexNames:
            for knowledgebaseId in knowledgebaseIds:
                table_name = f"{indexName}_{knowledgebaseId}"
                try:
                    select_clause = ", ".join(output)
                    where_clause = []
                    params = []
                    fulltext_sql = ''
                    vector_where = []
                    if filter_cond:
                        where_clause.append(filter_cond)
                    for matchExpr in matchExprs:
                        if isinstance(matchExpr, MatchTextExpr):
                            text_where = " AND ".join(where_clause) if where_clause else "1=1"
                            query_words = get_tsquery(matchExpr.matching_text)
                            text_topn = getattr(matchExpr, 'topn', 100)
                            fulltext_sql = generate_textsearch_query(matchExpr.fields, query_words, ", ".join(output), knowledgebaseId, table_name, text_topn, float(1.0 - vector_similarity_weight), text_where)
                        elif isinstance(matchExpr, MatchDenseExpr):
                            vector_score_expr = f"({matchExpr.vector_column_name} <=> %s::vector) " # cosine 
                            vector_where.append(f"score > {matchExpr.extra_options.get('threshold', 1.0)}")
                            vector = str(matchExpr.embedding_data)
                            params.append(vector)
                            params.append(vector)
                            vector_column_name = matchExpr.vector_column_name

                    if len(matchExprs) == 0:
                        query = f'select {", ".join(output)}, COUNT(*) OVER() AS total_count from {table_name} where {" AND ".join(where_clause) if where_clause else "1=1"} offset {offset} Limit {limit};'
                    elif len(vector_where) == 0 and fulltext_sql != '':
                        where_str = str(filter_cond)
                        query = f"""
                            set bm25_topk = 200;  
                            select /*+ indexscan({table_name} text_{knowledgebaseId}_content_with_weight)*/ {", ".join(output)}, COUNT(*) OVER() AS total_count
                            from {table_name}
                            where {where_str}
                            order by content_with_weight <&> '{query_words}' DESC
                            offset {offset} Limit {limit};
                            """
                        logging.info(f"query:{query}")
                    else:
                        # construct a complete query
                        query = base_query.format(
                            fulltext_sql=fulltext_sql,
                            fields=", ".join(output),
                            vector_score=vector_score_expr,
                            table_name=table_name,
                            condition=" AND ".join(where_clause) if where_clause else "1=1",
                            vector_condition=" OR ".join(vector_where) if vector_where else "1=1",
                            vector_similarity_weight=vector_similarity_weight,
                            vector_topn=next((e.topn for e in matchExprs if isinstance(e, MatchDenseExpr)), 1024)
                        )

                    # add pagination parameters
                    params.extend([offset, limit])

                    with self.conn.cursor() as cursor:
                        cursor.execute(query, params)
                        results = cursor.fetchall()
                        columns = [desc[0] for desc in cursor.description]
                        df = pd.DataFrame(results, columns=columns).drop_duplicates(subset = ['id']) # remove duplicates
                        if len(vector_where) != 0:
                            df[vector_column_name] = df[vector_column_name].apply(lambda x: list(map(float, ast.literal_eval(x))) if isinstance(x, str) else np.array([]))
                            total_hits_count += len(df)
                        else:
                            total_hits_count += results[0][-1] if results else 0
                        for col in df.select_dtypes(include=['object']).columns:
                            df[col] = df[col].fillna("")
                        df_list.append(df)
                        
                except Exception as e:
                    traceback.print_exc()
                    continue

        # combine results
        final_df = concat_dataframes(df_list, output)

        if matchExprs:
            if "score" not in final_df.columns:
                final_df["score"] = 0
            final_df['total_score'] = final_df["score"] + final_df[PAGERANK_FLD]
            final_df = final_df.sort_values(by='total_score', ascending=False)
            final_df = final_df.head(limit) 
            final_df = final_df.drop(columns=['total_score'])
        logger.info(f"final_df, total_hits_count: {len(final_df)} {total_hits_count}")
        return final_df, total_hits_count

    def get(
            self, chunkId: str, indexName: str, knowledgebaseIds: list[str]
    ) -> dict | None:
        df_list = []
        table_list = []
        self.check_reconnect()

        for knowledgebaseId in knowledgebaseIds:
            table_name = f"{indexName}_{knowledgebaseId}"
            table_list.append(table_name)

            query = f"SELECT * FROM {table_name} WHERE id = %s"
            logger.debug(f"Executing query on table: {table_name}")
            try:
                with self.conn.cursor() as cursor:
                    cursor.execute(query, (chunkId,))
                    data = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    if data:
                        df_list.append(pd.DataFrame(data, columns=columns))
            except psycopg2.Error as e:
                logger.warning(
                    f"Table not found or query failed: {table_name}, error: {str(e)}")
                continue

        if not df_list:
            logger.info(f"No data found for chunkId: {chunkId} in tables: {table_list}")
            return None

        res = concat_dataframes(df_list, ["id"])
        res_fields = self.getFields(res, res.columns.tolist())
        return res_fields.get(chunkId, None)

    def insert(
            self, documents: list[dict], indexName: str, knowledgebaseId: str = None
    ) -> list[str]:
        self.check_reconnect()

        table_name = f"{indexName}_{knowledgebaseId}"
        vector_size = 0
        patt = re.compile(r"q_(?P<vector_size>\d+)_vec")
        for k in documents[0].keys():
            m = patt.match(k)
            if m:
                vector_size = int(m.group("vector_size"))
                break
        if vector_size == 0:
            raise ValueError("Cannot infer vector size from documents")

        try:
            with self.conn.cursor() as cursor:
                cursor.execute(f"SELECT 1 FROM {table_name} LIMIT 1;")
        except psycopg2.Error:
            self.createIdx(indexName, knowledgebaseId, vector_size)

        docs = copy.deepcopy(documents)
        for d in docs:
            assert "_id" not in d
            assert "id" in d
            for k, v in d.items():
                if k in ["important_kwd", "question_kwd", "entities_kwd", "tag_kwd", "source_id"]:
                    assert isinstance(v, list)
                    d[k] = "###".join(v)
                    logger.info(f"insert_data: {d[k]}")
                elif re.search(r"_feas$", k):
                    d[k] = json.dumps(v)
                elif k == 'kb_id':
                    if isinstance(d[k], list):
                        d[k] = d[k][0]  
                elif k == "position_int":
                    assert isinstance(v, list)
                    arr = [num for row in v for num in row]
                    d[k] = "_".join(f"{num:08x}" for num in arr)
                elif k in ["page_num_int", "top_int"]:
                    assert isinstance(v, list)
                    d[k] = "_".join(f"{num:08x}" for num in v)

        # delete conflicting records
        ids = [f"'{d['id']}'" for d in docs]
        str_ids = ", ".join(ids)
        delete_query = f"DELETE FROM {table_name} WHERE id IN ({str_ids})"
        with self.conn.cursor() as cursor:
            cursor.execute(delete_query)

        longest_key_array = max(docs, key=lambda d: len(d.keys())).keys()
        insert_query = f"INSERT INTO {table_name} ({', '.join(longest_key_array)}) VALUES %s"
        values = [
            tuple(d.get(k, "") for k in longest_key_array)
            for d in docs
        ]
        from psycopg2.extras import execute_values
        with self.conn.cursor() as cursor:
            execute_values(cursor, insert_query, values)
        self.conn.commit()
        logger.debug(f"Inserted into {table_name}: {str_ids}.")

        vector_name = f"q_{vector_size}_vec"
        with self.conn.cursor() as cursor:
            # create vector index
            if self.enable_pq:
                pq_m = int(vector_size / 4)
                cursor.execute(sql.SQL("""
                                CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw({} vector_cosine_ops) WITH (m = 16, ef_construction = 64, enable_pq=on, pq_m={});
                            """).format(
                    sql.Identifier(f"q_vec_idx_{knowledgebaseId}"),
                    sql.Identifier(table_name),
                    sql.Identifier(vector_name),
                    sql.Literal(pq_m)
                ))
            else:
                cursor.execute(sql.SQL("""
                                CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw({} vector_cosine_ops) WITH (m = 16, ef_construction = 64);
                            """).format(
                    sql.Identifier(f"q_vec_idx_{knowledgebaseId}"),
                    sql.Identifier(table_name),
                    sql.Identifier(vector_name)
                ))
        self.conn.commit()
        return []

    def update(
            self, condition: dict, newValue: dict, indexName: str, knowledgebaseId: str
    ) -> bool:
        table_name = f"{indexName}_{knowledgebaseId}"
        table_columns = self.get_columns(table_name)
        filter_cond = equivalent_condition_to_str(condition, table_columns)

        update_set = []
        update_values = []
        for k, v in list(newValue.items()):
            if k in ["important_kwd", "question_kwd", "entities_kwd", "tag_kwd", "source_id"]:
                assert isinstance(v, list)
                v = "###".join(v)
            elif re.search(r"_feas$", k):
                v = json.dumps(v)
            elif k.endswith("_kwd") and isinstance(v, list):
                v = " ".join(v)
            elif k == 'kb_id' and isinstance(v, list):
                v = v[0]
            elif k == "position_int":
                assert isinstance(v, list)
                arr = [num for row in v for num in row]
                v = "_".join(f"{num:08x}" for num in arr)
            elif k in ["page_num_int", "top_int"]:
                assert isinstance(v, list)
                v = "_".join(f"{num:08x}" for num in v)
            elif k == "remove":
                del newValue[k]
                if v in ["PAGERANK_FLD"]:
                    newValue[v] = 0
                continue

            update_set.append(f"{k} = %s")
            update_values.append(v)

        update_query = f"""
            UPDATE {table_name}
            SET {', '.join(update_set)}
            WHERE {filter_cond}
            """

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(update_query, update_values)
                self.conn.commit()
                logger.debug(f"openGauss updated table {table_name}, filter {condition}, newValue {newValue}.")
                return True
            except psycopg2.Error as e:
                logger.error(f"Failed to update table {table_name}: {str(e)}")
                self.conn.rollback()
                return False


    def delete(self, condition: dict, indexName: str, knowledgebaseId: str) -> int:
        table_name = f"{indexName}_{knowledgebaseId}"

        if not self.indexExist(indexName, knowledgebaseId):
            logger.warning(f"Skipped deleting from table {table_name} since the table doesn't exist.")
            return 0

        table_columns = self.get_columns(table_name)
        filter_cond = equivalent_condition_to_str(condition, table_columns)

        delete_query = f"DELETE FROM {table_name} WHERE {filter_cond}"
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(delete_query)
                self.conn.commit()
                deleted_rows = cursor.rowcount
                logger.debug(
                    f"openGauss delete table {table_name}, filter {filter_cond}. Deleted rows: {deleted_rows}.")
                return deleted_rows
            except psycopg2.Error as e:
                logger.error(f"Failed to delete from table {table_name}: {str(e)}")
                self.conn.rollback()
                return 0


    """
    Helper functions for search result
    """

    def getTotal(self, res: tuple[pd.DataFrame, int] | pd.DataFrame) -> int:
        if isinstance(res, tuple):
            return res[1]
        return len(res)

    def getChunkIds(self, res: tuple[pd.DataFrame, int] | pd.DataFrame) -> list[str]:
        if isinstance(res, tuple):
            res = res[0]
        return list(res["id"])

    def getFields(self, res: tuple[pd.DataFrame, int] | pd.DataFrame, fields: list[str]) -> dict[str, dict]:
        logging.info(f"res fields : {res} {fields}")
        if isinstance(res, tuple):
            res = res[0]
        if not fields:
            return {}
        fieldsAll = fields.copy()
        fieldsAll.append('id')
        column_map = {col.lower(): col for col in res.columns}
        matched_columns = {column_map[col.lower()]: col for col in set(fieldsAll) if col.lower() in column_map}
        none_columns = [col for col in set(fieldsAll) if col.lower() not in column_map]

        res2 = res[matched_columns.keys()]
        res2 = res2.rename(columns=matched_columns)
        res2.drop_duplicates(subset=['id'], inplace=True)

        for column in res2.columns:
            k = column.lower()
            if k in ["important_kwd", "question_kwd", "entities_kwd", "tag_kwd", "source_id"]:
                res2[column] = res2[column].astype(str).fillna("")
                res2[column] = res2[column].apply(lambda v: [kwd for kwd in v.split("###") if kwd])
            elif k == "position_int":
                def to_position_int(v):
                    if v:
                        arr = [int(hex_val, 16) for hex_val in v.split('_')]
                        v = [arr[i:i + 5] for i in range(0, len(arr), 5)]
                    else:
                        v = []
                    return v

                res2[column] = res2[column].apply(to_position_int)
            elif k in ["page_num_int", "top_int"]:
                res2[column] = res2[column].apply(lambda v: [int(hex_val, 16) for hex_val in v.split('_')] if v else [])
            else:
                pass
        for column in none_columns:
            res2[column] = None

        return res2.set_index("id").to_dict(orient="index")
    
    def get_columns(self, table_name: str):
        table_name = table_name[:-10] + "%"
        table_columns = {}
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    f"SELECT column_name, data_type, column_default FROM information_schema.columns WHERE table_name like %s",
                    (table_name,))
                for row in cursor.fetchall():
                    column_name, data_type, column_default = row
                    table_columns[column_name] = (data_type, column_default)
            return table_columns
        except psycopg2.Error as e:
            logger.error(f"Failed to get columns from table {table_name}: {str(e)}")
            self.conn.rollback()
            return table_columns
        
    def check_reconnect(self):
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT 1")
        except Exception as e:
            logging.info("Connection lost,reconnecting...")
            self.__init__()

    def getHighlight(self, res: tuple[pd.DataFrame, int] | pd.DataFrame, keywords: list[str], fieldnm: str):
        if isinstance(res, tuple):
            res = res[0]
        ans = {}
        num_rows = len(res)
        column_id = res["id"]
        if fieldnm not in res:
            return {}
        pattern = re.compile("|".join(map(re.escape, keywords)))
        for i in range(num_rows):
            id = column_id.iloc[i]

            txt = res[fieldnm].iloc[i]
            txt = re.sub(r"[\r\n]", " ", txt, flags=re.IGNORECASE | re.MULTILINE)
            txts = []
            for t in re.split(r"[.?!;\n]", txt):
                if not t.strip(): 
                    continue
                for w in keywords:
                    t = pattern.sub(lambda m: f"<em>{m.group(0)}</em>", t)
                if not re.search(r"<em>[^<>]+</em>", t, flags=re.IGNORECASE | re.MULTILINE):
                    continue
                txts.append(t)
            ans[id] = "...".join(txts) if txts else txt[:100] + "..." 
        return ans

    def getAggregation(self, res: tuple[pd.DataFrame, int] | pd.DataFrame, fieldnm: str):
        """
        TODO
        """
        return list()

    """
    SQL
    """

    def sql(sql: str, fetch_size: int, format: str):
        raise NotImplementedError("Not implemented")