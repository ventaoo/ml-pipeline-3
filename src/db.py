import os
import pandas as pd
from sqlalchemy import create_engine, text
from typing import List, Tuple

from init_env import decrypt_with_ansible_lib

encrypted_file = os.getenv("DECRYPT_FILE_PATH")
encrypted_password = os.getenv("DECRYPT_PASSWORD")
output_file = os.getenv("OUTPUT_FILE")
decrypt_with_ansible_lib(encrypted_file, encrypted_password, output_file)

def create_connection():
    """
    使用SQLAlchemy创建数据库连接
    """
    db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(db_url)
    return engine

def get_data_from_db(query: str) -> pd.DataFrame:
    """
    从数据库获取数据并返回为DataFrame
    """
    engine = create_connection()
    # 使用SQLAlchemy的引擎创建连接，并通过pandas读取SQL查询结果
    with engine.connect() as conn:
        data = pd.read_sql(text(query), conn)
    return data

def store_prediction_results(results: List[Tuple[str, int]], table_name: str) -> None:
    """
    将模型预测结果存储到数据库中
    :param results: 模型预测结果的列表，每个元素为一个 (输入文本, 预测值) 元组
    :param table_name: 表的名称，用于插入数据
    """
    engine = create_connection()
    try:
        with engine.connect() as conn:
            insert_query = f"""
            INSERT INTO {table_name} (input_text, prediction_value)
            VALUES (:input_text, :prediction_value)
            """
            for input_text, prediction in results:
                conn.execute(text(insert_query), {
                    "input_text": input_text,
                    "prediction_value": prediction
                })
            conn.commit()
        print(f"Successfully inserted {len(results)} records into {table_name}.")
    except Exception as e:
        print(f"Error while inserting predictions: {e}")

    
def get_csv_data(file_path: str) -> pd.DataFrame:
    """
    从CSV文件加载数据
    """
    return pd.read_csv(file_path)

def load_csv_to_db(file_path: str, table_name: str) -> None:
    """
    从CSV文件加载数据并将其插入到数据库中的指定表
    """
    # 加载CSV数据
    data = get_csv_data(file_path)

    # 检查数据是否为空
    if data.empty:
        print("The CSV file is empty.")
        return

    # 创建数据库连接
    engine = create_connection()

    try:
        # 将数据插入数据库中的指定表
        data.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Data from {file_path} successfully inserted into {table_name}.")
        return True
    except Exception as e:
        print(f"Error while inserting data from CSV: {e}")
    
    return False


if __name__ == '__main__':
    # Init database.
    init_sql_path = './src/init_db.sql'
    with create_connection().connect() as conn:
        with open(init_sql_path) as f:
            conn.execute(text(f.read()))
        conn.commit()
    print(f'Database init -> {init_sql_path}.')
    
    # Load csv to database
    load_csv_to_db('./data/JEOPARDY_CSV.csv', 'jeopardy')
    print('Load the csv data to the database...')
