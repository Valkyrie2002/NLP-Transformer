import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path


class HistoryDatabase:
    """Quản lý lịch sử phân loại cảm xúc với SQLite"""
    
    def __init__(self, db_path="./data/history.db"):
        self.db_path = db_path
        # Tạo thư mục data nếu chưa tồn tại
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()
    
    def init_db(self):
        """Khởi tạo database và bảng history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence_score REAL NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, text, predicted_label, confidence_score):
        """Lưu một dự đoán mới vào database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute('''
            INSERT INTO history (timestamp, text, predicted_label, confidence_score)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, text, predicted_label, confidence_score))
        
        conn.commit()
        conn.close()
    
    def get_history(self, limit=None):
        """Lấy lịch sử phân loại (mới nhất trước)
        
        Args:
            limit: Số lượng bản ghi tối đa (None = tất cả)
            
        Returns:
            List of tuples: (id, timestamp, text, predicted_label, confidence_score)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if limit:
            cursor.execute('''
                SELECT id, timestamp, text, predicted_label, confidence_score
                FROM history
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
        else:
            cursor.execute('''
                SELECT id, timestamp, text, predicted_label, confidence_score
                FROM history
                ORDER BY timestamp DESC
            ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def get_count(self):
        """Đếm tổng số bản ghi trong lịch sử"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM history')
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    
    def clear_history(self):
        """Xóa toàn bộ lịch sử"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM history')
        
        conn.commit()
        conn.close()
    
    def export_to_csv(self, output_path="history_export.csv"):
        """Xuất lịch sử ra file CSV
        
        Args:
            output_path: Đường dẫn file CSV đầu ra
            
        Returns:
            Path to the exported CSV file
        """
        conn = sqlite3.connect(self.db_path)
        
        # Đọc dữ liệu vào DataFrame
        df = pd.read_sql_query('''
            SELECT timestamp, text, predicted_label, confidence_score
            FROM history
            ORDER BY timestamp DESC
        ''', conn)
        
        conn.close()
        
        # Xuất ra CSV
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        return output_path
