import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import os  
from datetime import datetime
import json 
from dotenv import load_dotenv

class YouTubeDBSetup:
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
        -- 타임존 설정
        SET timezone = 'Asia/Seoul';
        
        -- 1. Channel 테이블 생성
        CREATE TABLE IF NOT EXISTS channel (
            channel_id VARCHAR NOT NULL PRIMARY KEY,
            customUrl VARCHAR,
            title VARCHAR,
            country VARCHAR,
            description VARCHAR,
            published_at VARCHAR,
            etag VARCHAR,
            hiddenSubscriberCount BOOLEAN,
            subscriberCount BIGINT,
            videoCount BIGINT,
            viewCount BIGINT,
            thumbnail_url VARCHAR,
            collection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 2. Videos 테이블 생성
        CREATE TABLE IF NOT EXISTS videos (
            video_id VARCHAR NOT NULL PRIMARY KEY,
            title VARCHAR,
            channel_title VARCHAR,
            channel_id VARCHAR,
            published_at VARCHAR,
            category_id VARCHAR,
            view_count BIGINT,
            like_count BIGINT,
            comment_count BIGINT,
            duration_formatted BIGINT,
            made_for_kids BOOLEAN,
            tags VARCHAR,
            description VARCHAR,
            script VARCHAR,
            script_timestamp VARCHAR,
            collection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            like_ratio FLOAT,
            engagement_rate FLOAT,
            thumbnail_maxres VARCHAR
            
        );

        -- 3. Comments 테이블 생성
        CREATE TABLE IF NOT EXISTS comments (
            comment_id VARCHAR NOT NULL PRIMARY KEY,
            video_id VARCHAR,
            author VARCHAR,
            author_channel_id VARCHAR,
            comment_text VARCHAR,
            like_count BIGINT,
            published_at VARCHAR,
            updated_at VARCHAR,
            reply_count BIGINT,
            is_reply BOOLEAN,
            parent_id VARCHAR,
            collection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            -- 새로 추가된 혐오표현 분석 칼럼들
            is_hate_speech BOOLEAN,
            categories CHARACTER VARYING[],
            similar_cases_used CHARACTER VARYING[],
            target_group CHARACTER VARYING,
            hate_type CHARACTER VARYING,
            used_prompt CHARACTER VARYING
        );
        """
        
        # 인덱스 생성 SQL
        create_indexes_sql = """
        -- 인덱스 생성 (성능 최적화)
        CREATE INDEX IF NOT EXISTS idx_videos_channel_id ON videos(channel_id);
        CREATE INDEX IF NOT EXISTS idx_comments_video_id ON comments(video_id);
        CREATE INDEX IF NOT EXISTS idx_comments_parent_id ON comments(parent_id);
        CREATE INDEX IF NOT EXISTS idx_videos_published_at ON videos(published_at);
        CREATE INDEX IF NOT EXISTS idx_comments_published_at ON comments(published_at);
        """
        
        try:
            # 데이터베이스 연결
            connection = psycopg2.connect(**self.db_config)
            cursor = connection.cursor()
            
            print("PostgreSQL 연결 성공!")
            
            # 테이블 생성
            cursor.execute(create_tables_sql)
            print("테이블 생성 완료!")
            
            # 인덱스 생성
            cursor.execute(create_indexes_sql)
            print("인덱스 생성 완료!")
            
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

    def save_channel_data(self, channel_data):
        """채널 데이터 저장"""
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            cursor = connection.cursor()
            
            # 단일 채널 데이터인 경우 리스트로 변환
            if isinstance(channel_data, dict):
                channel_data = [channel_data]
            
            # INSERT SQL (ON CONFLICT로 중복 처리)
            insert_sql = """
                INSERT INTO channel (
                    channel_id, customUrl, title, country, description, 
                    published_at, etag, hiddenSubscriberCount, subscriberCount, 
                    videoCount, viewCount, thumbnail_url
                ) VALUES %s
                ON CONFLICT (channel_id) 
                DO UPDATE SET
                    customUrl = EXCLUDED.customUrl,
                    title = EXCLUDED.title,
                    country = EXCLUDED.country,
                    description = EXCLUDED.description,
                    published_at = EXCLUDED.published_at,
                    etag = EXCLUDED.etag,
                    hiddenSubscriberCount = EXCLUDED.hiddenSubscriberCount,
                    subscriberCount = EXCLUDED.subscriberCount,
                    videoCount = EXCLUDED.videoCount,
                    viewCount = EXCLUDED.viewCount,
                    thumbnail_url = EXCLUDED.thumbnail_url,
                    collection_time = CURRENT_TIMESTAMP;
            """
            
            # 데이터 준비
            values = []
            for channel in channel_data:
                values.append((
                    channel.get('channel_id'),
                    channel.get('customUrl'),
                    channel.get('title'),
                    channel.get('country'),
                    channel.get('description'),
                    channel.get('published_at'),
                    channel.get('etag'),
                    channel.get('hiddenSubscriberCount'),
                    channel.get('subscriberCount'),
                    channel.get('videoCount'),
                    channel.get('viewCount'),
                    channel.get('thumbnail_url')
                ))
            
            # 데이터 삽입
            execute_values(cursor, insert_sql, values)
            connection.commit()
            
            print(f"✅ 채널 데이터 {len(values)}개 저장 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 채널 데이터 저장 실패: {e}")
            connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
                
    def save_video_data(self, video_data):
        """영상 데이터 저장"""
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            cursor = connection.cursor()
            
            # 단일 영상 데이터인 경우 리스트로 변환
            if isinstance(video_data, dict):
                video_data = [video_data]
            
            # INSERT SQL
            insert_sql = """
                INSERT INTO videos (
                    video_id, title, channel_title, channel_id, published_at,
                    category_id, view_count, like_count, comment_count,
                    duration_formatted, tags,
                    script, script_timestamp,description, like_ratio, engagement_rate, thumbnail_maxres
                ) VALUES %s
                ON CONFLICT (video_id)
                DO UPDATE SET
                    title = EXCLUDED.title,
                    channel_title = EXCLUDED.channel_title,
                    channel_id = EXCLUDED.channel_id,
                    published_at = EXCLUDED.published_at,
                    category_id = EXCLUDED.category_id,
                    view_count = EXCLUDED.view_count,
                    like_count = EXCLUDED.like_count,
                    comment_count = EXCLUDED.comment_count,
                    duration_formatted = EXCLUDED.duration_formatted,
                    tags = EXCLUDED.tags,
                    description = EXCLUDED.description,
                    script = EXCLUDED.script,
                    script_timestamp = EXCLUDED.script_timestamp,
                    like_ratio = EXCLUDED.like_ratio,
                    engagement_rate = EXCLUDED.engagement_rate,
                    thumbnail_maxres = EXCLUDED.thumbnail_maxres,
                    collection_time = CURRENT_TIMESTAMP;
            """
            
            # 데이터 준비
            values = []
            for video in video_data:
                # tags가 리스트인 경우 문자열로 변환
                tags = video.get('tags')
                if isinstance(tags, list):
                    tags = ', '.join(tags)
                
                values.append((
                    video.get('video_id'),
                    video.get('title'),
                    video.get('channel_title'),
                    video.get('channel_id'),
                    video.get('published_at'),
                    video.get('category_id'),
                    video.get('view_count'),
                    video.get('like_count'),
                    video.get('comment_count'),
                    video.get('duration_formatted'),
                    tags,
                    video.get('script'),
                    video.get('script_timestamp'),
                    video.get('description'),
                    video.get('like_ratio'),
                    video.get('engagement_rate'),
                    video.get('thumbnail_maxres')
                ))
            
            # 데이터 삽입
            execute_values(cursor, insert_sql, values)
            connection.commit()
            
            print(f"✅ 영상 데이터 {len(values)}개 저장 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 영상 데이터 저장 실패: {e}")
            connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
                
    def save_comment_data(self, comment_data):
        """댓글 데이터 저장"""
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            cursor = connection.cursor()
            
            # 단일 댓글 데이터인 경우 리스트로 변환
            if isinstance(comment_data, dict):
                comment_data = [comment_data]
            
            # INSERT SQL
            insert_sql = """
                INSERT INTO comments (
                    comment_id, video_id, author, author_channel_id, 
                    comment_text, like_count, published_at, updated_at,
                    reply_count, is_reply, parent_id
                ) VALUES %s
                ON CONFLICT (comment_id)
                DO UPDATE SET
                    video_id = EXCLUDED.video_id,
                    author = EXCLUDED.author,
                    author_channel_id = EXCLUDED.author_channel_id,
                    comment_text = EXCLUDED.comment_text,
                    like_count = EXCLUDED.like_count,
                    published_at = EXCLUDED.published_at,
                    updated_at = EXCLUDED.updated_at,
                    reply_count = EXCLUDED.reply_count,
                    is_reply = EXCLUDED.is_reply,
                    parent_id = EXCLUDED.parent_id,
                    collection_time = CURRENT_TIMESTAMP;
            """
            
            # 데이터 준비
            values = []
            for comment in comment_data:
                values.append((
                    comment.get('comment_id'),
                    comment.get('video_id'),
                    comment.get('author'),
                    comment.get('author_channel_id'),
                    comment.get('comment_text'),
                    comment.get('like_count'),
                    comment.get('published_at'),
                    comment.get('updated_at'),
                    comment.get('reply_count'),
                    comment.get('is_reply'),
                    comment.get('parent_id')
                ))
            
            # 데이터 삽입
            execute_values(cursor, insert_sql, values)
            connection.commit()
            
            print(f"✅ 댓글 데이터 {len(values)}개 저장 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 댓글 데이터 저장 실패: {e}")
            connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
                
    def update_hate_speech_analysis(self, comment_id, analysis_result):
        """혐오표현 분석 결과 업데이트"""
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            cursor = connection.cursor()
            
            update_sql = """
                UPDATE comments 
                SET 
                    is_hate_speech = %s,
                    categories = %s,
                    similar_cases_used = %s,
                    target_group = %s,
                    hate_type = %s,
                    used_prompt = %s
                WHERE comment_id = %s;
            """
            
            cursor.execute(update_sql, (
                analysis_result.get('is_hate_speech'),
                analysis_result.get('categories'),  # 배열
                analysis_result.get('similar_cases_used'),  # 배열
                analysis_result.get('target_group'),
                analysis_result.get('hate_type'),
                analysis_result.get('used_prompt'),
                comment_id
            ))
            
            connection.commit()
            return True
            
        except Exception as e:
            print(f"❌ 혐오표현 분석 결과 업데이트 실패: {e}")
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
            
            cursor.execute("SELECT COUNT(*) FROM channel;")
            channel_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM videos;")
            video_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM comments;")
            comment_count = cursor.fetchone()[0]
            
            print(f"📊 저장된 데이터:")
            print(f"   채널: {channel_count:,}개")
            print(f"   영상: {video_count:,}개")
            print(f"   댓글: {comment_count:,}개")
            
            return {
                'channels': channel_count,
                'videos': video_count,
                'comments': comment_count
            }
            
        except Exception as e:
            print(f"❌ 데이터 개수 조회 실패: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
                
    def get_unique_channel_ids(self):
        """videos 테이블에서 중복 없는 channel_id 목록 조회"""
        connection = self.get_connection()
        if not connection:
            return None
        
        try:
            cursor = connection.cursor()
            
            # 중복 없는 channel_id 조회
            cursor.execute("""
                SELECT DISTINCT channel_id 
                FROM videos 
                WHERE channel_id IS NOT NULL
                ORDER BY channel_id;
            """)
            
            results = cursor.fetchall()
            
            # 튜플을 리스트로 변환
            channel_ids = [row[0] for row in results]
            
            print(f"📋 고유 채널 ID: {len(channel_ids):,}개")
            
            return channel_ids
            
        except Exception as e:
            print(f"❌ 채널 ID 조회 실패: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    def get_unique_video_ids(self):
        """comments 테이블에서 중복 없는 video_id 목록 조회"""
        connection = self.get_connection()
        if not connection:
            return None
        
        try:
            cursor = connection.cursor()
            
            # 중복 없는 video_id 조회
            cursor.execute("""
                SELECT DISTINCT video_id 
                FROM comments 
                WHERE video_id IS NOT NULL
                ORDER BY video_id;
            """)
            
            results = cursor.fetchall()
            
            # 튜플을 리스트로 변환
            video_ids = [row[0] for row in results]
            
            print(f"📋 고유 비디오 ID: {len(video_ids):,}개")
            
            return video_ids
            
        except Exception as e:
            print(f"❌ 비디오 ID 조회 실패: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def get_comments_generator(self, batch_size=1000, include_analysis=True):
        """Comments table의 요소를 제너레이터로 리턴"""
        
        offset = 0
        
        while True:
            connection = self.get_connection()
            if not connection:
                break;
        
            try:
                cursor = connection.cursor()
                columns = """
                    comment_id, video_id, author, author_channel_id, 
                    comment_text, like_count, published_at, updated_at,
                    reply_count, is_reply, parent_id, collection_time,
                    is_hate_speech, categories, similar_cases_used,
                    target_group, hate_type, used_prompt
                """
                query = f"""
                SELECT {columns}
                FROM comments 
                ORDER BY comment_id  -- 변경되지 않는 컬럼으로 정렬
                LIMIT %s OFFSET %s;
                """
                cursor.execute(query, (batch_size, offset))
                rows = cursor.fetchall()
                for row in rows:
                    comment = {
                        'comment_id': row[0],
                        'video_id': row[1],
                        'author': row[2],
                        'author_channel_id': row[3],
                        'comment_text': row[4],
                        'like_count': row[5],
                        'published_at': row[6],
                        'updated_at': row[7],
                        'reply_count': row[8],
                        'is_reply': row[9],
                        'parent_id': row[10],
                        'collection_time': row[11]
                    }
                    
                    if include_analysis and len(row) > 12:
                        comment.update({
                            'is_hate_speech': row[12],
                            'categories': row[13],
                            'similar_cases_used': row[14],
                            'target_group': row[15],
                            'hate_type': row[16],
                            'used_prompt': row[17]
                        })
                    
                    yield comment
            
            except Exception as e:
                print(f"배치 {offset // batch_size + 1} 처리 실패: {e}")
                break
            
            finally:
                if cursor:
                    cursor.close()
                if connection:
                    connection.close()

    def get_comments_by_video_id(self, video_id, verbose=True, include_analysis=True):
        """특정 video_id에 대한 댓글 조회 (분석 결과 포함)"""
        connection = self.get_connection()
        if not connection:
            return None
        
        try:
            cursor = connection.cursor()
            
            if include_analysis:
                # 분석 결과도 함께 조회
                cursor.execute("""
                    SELECT 
                        comment_id, video_id, author, author_channel_id, 
                        comment_text, like_count, published_at, updated_at,
                        reply_count, is_reply, parent_id, collection_time,
                        is_hate_speech, categories, similar_cases_used,
                        target_group, hate_type, used_prompt
                    FROM comments 
                    WHERE video_id = %s
                    ORDER BY published_at DESC;
                """, (video_id,))
            else:
                # 기존과 동일
                cursor.execute("""
                    SELECT * 
                    FROM comments 
                    WHERE video_id = %s
                    ORDER BY published_at DESC;
                """, (video_id,))
            
            results = cursor.fetchall()
            
            # 결과를 딕셔너리 형태로 변환
            comments = []
            for row in results:
                comment = {
                    'comment_id': row[0],
                    'video_id': row[1],
                    'author': row[2],
                    'author_channel_id': row[3],
                    'comment_text': row[4],
                    'like_count': row[5],
                    'published_at': row[6],
                    'updated_at': row[7],
                    'reply_count': row[8],
                    'is_reply': row[9],
                    'parent_id': row[10],
                    'collection_time': row[11]
                }
                
                if include_analysis and len(row) > 12:
                    comment.update({
                        'is_hate_speech': row[12],
                        'categories': row[13],
                        'similar_cases_used': row[14],
                        'target_group': row[15],
                        'hate_type': row[16],
                        'used_prompt': row[17]
                    })
                
                comments.append(comment)
            
            if verbose:
                print(f"📋 {len(comments):,}개의 댓글 조회 완료!")
            return comments
            
        except Exception as e:
            if verbose:
                print(f"❌ 댓글 조회 실패: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
        
    def get_video_metadata(self, video_id, verbose=True):
        """특정 video_id에 대한 메타데이터 조회"""
        connection = self.get_connection()
        if not connection:
            return None
        
        try:
            cursor = connection.cursor()
            
            # video_id에 해당하는 메타데이터 조회
            cursor.execute("""
                SELECT * 
                FROM videos 
                WHERE video_id = %s;
            """, (video_id,))
            
            result = cursor.fetchone()
            
            if result:
                metadata = {
                    'video_id': result[0],
                    'title': result[1],
                    'channel_title': result[2],
                    'channel_id': result[3],
                    'published_at': result[4],
                    'category_id': result[5],
                    'view_count': result[6],
                    'like_count': result[7],
                    'comment_count': result[8],
                    'duration_formatted': result[9],
                    'tags': result[10],
                    'script': result[11],
                    'script_timestamp': result[12],
                    'description': result[13],
                    'like_ratio': result[14],
                    'engagement_rate': result[15],
                    'thumbnail_maxres': result[16]
                }
                if (verbose):
                    print(f"📋 {video_id}에 대한 메타데이터 조회 완료!")
                return metadata
            else:
                if (verbose):
                    print(f"❌ {video_id}에 대한 메타데이터가 없습니다.")
                return None
            
        except Exception as e:
            if (verbose):
                print(f"❌ 메타데이터 조회 실패: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def get_unanalyzed_comments(self, limit=100):
        """아직 분석되지 않은 댓글들 조회"""
        connection = self.get_connection()
        if not connection:
            return None
        
        try:
            cursor = connection.cursor()
            
            cursor.execute("""
                SELECT comment_id, comment_text, video_id, author
                FROM comments 
                WHERE is_hate_speech IS NULL
                ORDER BY collection_time DESC
                LIMIT %s;
            """, (limit,))
            
            results = cursor.fetchall()
            
            comments = []
            for row in results:
                comments.append({
                    'comment_id': row[0],
                    'comment_text': row[1],
                    'video_id': row[2],
                    'author': row[3]
                })
            
            return comments
            
        except Exception as e:
            print(f"❌ 미분석 댓글 조회 실패: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()


# 사용 예시
if __name__ == "__main__":
    
    # DB 셋업 인스턴스 생성
    db_setup = YouTubeDBSetup()
    
    # 테이블 생성
    db_setup.create_tables()
    
    print("YouTube 데이터베이스 셋업 완료!")