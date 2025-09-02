import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import os  
from datetime import datetime
import json 
from dotenv import load_dotenv

class HateSpeechDBSetup:
    def __init__(self):
        load_dotenv()
        # 데이터베이스 연결 정보
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'port': os.getenv('DB_PORT', '5432')
        }
    
    def create_tables(self):
        """테이블 생성 및 초기 설정"""
        
        # 테이블 생성 SQL
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS scriptresult (
            video_id VARCHAR NOT NULL,
            script_index INT NOT NULL,
            input_text VARCHAR,
            is_hate_speech BOOLEAN,
            categories VARCHAR[],
            evidence_strength FLOAT,
            reasoning VARCHAR,
            similar_cases_used VARCHAR[],
            target_group VARCHAR,
            hate_type VARCHAR,
            used_prompt VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (video_id, script_index)
        );
        """
        
        try:
            # 데이터베이스 연결
            connection = psycopg2.connect(**self.db_config)
            cursor = connection.cursor()
            
            print("PostgreSQL 연결 성공!")
            
            # 테이블 생성
            cursor.execute(create_tables_sql)
            print("테이블 생성 완료!")
            
            # 변경사항 커밋
            connection.commit()
            
            # 테이블 확인
            self.check_tables(cursor)
            
        except psycopg2.Error as e:
            print(f"데이터베이스 에러: {e}")
        except Exception as e:
            print(f"일반 에러: {e}")
        finally:
            # 연결 종료
            if cursor:
                cursor.close()
            if connection:
                connection.close()
            print("데이터베이스 연결 종료")
    
    def check_tables(self, cursor):
        """생성된 테이블 확인"""
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        print("\n생성된 테이블:")
        for table in tables:
            print(f"- {table[0]}")
    
    def get_connection(self):
        """데이터베이스 연결 반환 (다른 모듈에서 사용)"""
        try:
            return psycopg2.connect(**self.db_config)
        except psycopg2.Error as e:
            print(f"연결 실패: {e}")
            return None
        
    def get_next_script_index(self, cursor, video_id):
        """해당 video_id의 다음 script_index 반환"""
        cursor.execute("""
            SELECT COALESCE(MAX(script_index), -1) + 1 
            FROM scriptresult 
            WHERE video_id = %s
        """, (video_id,))
        
        result = cursor.fetchone()
        return result[0] if result else 0

    def find_existing_script_by_text(self, cursor, video_id, input_text):
        """input_text를 기준으로 기존 데이터 찾기"""
        cursor.execute("""
            SELECT script_index 
            FROM scriptresult 
            WHERE video_id = %s AND input_text = %s
        """, (video_id, input_text))
        
        result = cursor.fetchone()
        return result[0] if result else None

    def save_script_result_manual(self, script_data):
        """스크립트 분석 결과 저장"""
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            cursor = connection.cursor()
            
            # 단일 스크립트 데이터인 경우 리스트로 변환
            if isinstance(script_data, dict):
                script_data = [script_data]
            
            # INSERT SQL
            insert_sql = """
                INSERT INTO scriptresult (
                    video_id, script_index, input_text, is_hate_speech, categories,
                    evidence_strength, reasoning, similar_cases_used, target_group,
                    hate_type, used_prompt
                ) VALUES %s
                ON CONFLICT (video_id, script_index) 
                DO UPDATE SET
                    input_text = EXCLUDED.input_text,
                    is_hate_speech = EXCLUDED.is_hate_speech,
                    categories = EXCLUDED.categories,
                    evidence_strength = EXCLUDED.evidence_strength,
                    reasoning = EXCLUDED.reasoning,
                    similar_cases_used = EXCLUDED.similar_cases_used,
                    target_group = EXCLUDED.target_group,
                    hate_type = EXCLUDED.hate_type,
                    used_prompt = EXCLUDED.used_prompt,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            # 데이터 준비
            values = []
            for script in script_data:
                values.append((
                    script.get('video_id'),
                    script.get('script_index'),
                    script.get('input_text'),
                    script.get('is_hate_speech'),
                    script.get('categories'),
                    script.get('evidence_strength'),
                    script.get('reasoning'),
                    script.get('similar_cases_used'),
                    script.get('target_group'),
                    script.get('hate_type'),
                    script.get('used_prompt')
                ))
            
            # 데이터 삽입
            execute_values(cursor, insert_sql, values)
            connection.commit()
            
            print(f"✅ 스크립트 분석 결과 {len(values)}개 저장 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 스크립트 분석 결과 저장 실패: {e}")
            connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
                
    def get_data_count(self):
        """저장된 데이터 개수 확인"""
        connection = self.get_connection()
        if not connection:
            return None
        
        try:
            cursor = connection.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM scriptresult;")
            script_result_count = cursor.fetchone()[0]
            
            print(f"📊 저장된 데이터:")
            print(f"   스크립트 분석 결과: {script_result_count:,}개")
            
            return {
                'scriptresult': script_result_count
            }
            
        except Exception as e:
            print(f"❌ 데이터 개수 조회 실패: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
                
    def save_script(self, script_data, video_id : str, used_prompt : str):
        result = {}
        result['video_id'] = video_id
        result['input_text'] = script_data.input_text
        result['is_hate_speech'] = script_data.is_hate_speech
        result['categories'] = script_data.categories
        result['evidence_strength'] = script_data.evidence_strength
        result['reasoning'] = script_data.reasoning
        result['similar_cases_used'] = script_data.similar_cases_used
        result['target_group'] = script_data.target_group
        result['hate_type'] = script_data.hate_type
        result['used_prompt'] = used_prompt
        self.save_script_result(result)
        
        
        
        
        
    def save_script_result(self, script_data):
        """스크립트 분석 결과 저장 - input_text 기준 중복 체크 및 갱신"""
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            cursor = connection.cursor()
            
            # 단일 스크립트 데이터인 경우 리스트로 변환
            if isinstance(script_data, dict):
                script_data = [script_data]
            
            # video_id별로 그룹화
            video_groups = {}
            for script in script_data:
                video_id = script.get('video_id')
                if video_id not in video_groups:
                    video_groups[video_id] = []
                video_groups[video_id].append(script)
            
            # 각각 처리 (배치 처리 대신 개별 처리로 변경)
            total_inserted = 0
            total_updated = 0
            
            for video_id, scripts in video_groups.items():
                for script in scripts:
                    input_text = script.get('input_text')
                    
                    # 기존 데이터 확인 (input_text 기준)
                    existing_index = self.find_existing_script_by_text(cursor, video_id, input_text)
                    
                    if existing_index is not None:
                        # 기존 데이터 업데이트
                        update_sql = """
                            UPDATE scriptresult 
                            SET 
                                is_hate_speech = %s,
                                categories = %s,
                                evidence_strength = %s,
                                reasoning = %s,
                                similar_cases_used = %s,
                                target_group = %s,
                                hate_type = %s,
                                used_prompt = %s,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE video_id = %s AND script_index = %s
                        """
                        
                        cursor.execute(update_sql, (
                            script.get('is_hate_speech'),
                            script.get('categories'),
                            script.get('evidence_strength'),
                            script.get('reasoning'),
                            script.get('similar_cases_used'),
                            script.get('target_group'),
                            script.get('hate_type'),
                            script.get('used_prompt'),
                            video_id,
                            existing_index
                        ))
                        
                        total_updated += 1
                        print(f"🔄 {video_id}[{existing_index}] 기존 데이터 업데이트")
                        
                    else:
                        # 새 데이터 삽입 - 자동 인덱스 할당
                        next_index = self.get_next_script_index(cursor, video_id)
                        
                        insert_sql = """
                            INSERT INTO scriptresult (
                                video_id, script_index, input_text, is_hate_speech, categories,
                                evidence_strength, reasoning, similar_cases_used, target_group,
                                hate_type, used_prompt
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        
                        cursor.execute(insert_sql, (
                            video_id,
                            next_index,
                            input_text,
                            script.get('is_hate_speech'),
                            script.get('categories'),
                            script.get('evidence_strength'),
                            script.get('reasoning'),
                            script.get('similar_cases_used'),
                            script.get('target_group'),
                            script.get('hate_type'),
                            script.get('used_prompt')
                        ))
                        
                        total_inserted += 1
                        print(f"📝 {video_id}[{next_index}] 새 데이터 삽입")
            
            connection.commit()
            
            print(f"✅ 완료 - 신규: {total_inserted}개, 업데이트: {total_updated}개")
            return True
            
        except Exception as e:
            print(f"❌ 스크립트 분석 결과 저장 실패: {e}")
            connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
                
# 사용 예시
if __name__ == "__main__":
    
    # DB 셋업 인스턴스 생성
    db_setup = HateSpeechDBSetup()
    
    # 테이블 생성
    db_setup.create_tables()
    
    print("scriptresult 데이터베이스 셋업 완료!")