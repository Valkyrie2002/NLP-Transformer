"""
Script để chuyển đổi dataset từ 6 nhãn cảm xúc sang 3 nhãn sentiment cơ bản.

Mapping:
- POSITIVE ← JOY, POSITIVE, SURPRISE
- NEGATIVE ← ANGER, FEAR, SADNESS, NEGATIVE
- NEUTRAL ← NEUTRAL, OTHER
"""

import pandas as pd
import os
from pathlib import Path

# Mapping từ 6 nhãn sang 3 nhãn
LABEL_MAPPING = {
    # Positive emotions
    'JOY': 'POSITIVE',
    'POSITIVE': 'POSITIVE',
    'SURPRISE': 'POSITIVE',
    
    # Negative emotions
    'ANGER': 'NEGATIVE',
    'FEAR': 'NEGATIVE',
    'SADNESS': 'NEGATIVE',
    'NEGATIVE': 'NEGATIVE',
    
    # Neutral
    'NEUTRAL': 'NEUTRAL',
    'OTHER': 'NEUTRAL'
}

def convert_dataset():
    """Chuyển đổi dataset từ 6 nhãn sang 3 nhãn"""
    
    # Tạo thư mục output nếu chưa tồn tại
    output_dir = Path('../data/processed_3labels')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Bắt đầu chuyển đổi dataset từ 6 nhãn sang 3 nhãn...\n")
    
    # Chuyển đổi từng split
    for split in ['train', 'val', 'test']:
        input_path = f'../data/processed/{split}.csv'
        output_path = output_dir / f'{split}.csv'
        
        print(f"📂 Xử lý {split}.csv...")
        
        # Đọc file
        df = pd.read_csv(input_path, encoding='utf-8')
        original_count = len(df)
        
        # Hiển thị phân bố 6 nhãn
        print(f"  Phân bố 6 nhãn gốc:")
        for label, count in df['emotion'].value_counts().items():
            print(f"    {label}: {count}")
        
        # Chuyển đổi nhãn
        df['emotion'] = df['emotion'].map(LABEL_MAPPING)
        
        # Kiểm tra có missing values không (do nhãn không khớp)
        if df['emotion'].isna().any():
            print(f"  ⚠️ Cảnh báo: Có {df['emotion'].isna().sum()} nhãn không khớp mapping!")
            df = df.dropna(subset=['emotion'])
        
        # Hiển thị phân bố 3 nhãn mới
        print(f"  Phân bố 3 nhãn mới:")
        for label, count in df['emotion'].value_counts().items():
            print(f"    {label}: {count}")
        
        # Lưu file
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"  ✅ Đã lưu: {output_path} ({len(df)} samples)\n")
    
    print("✅ Hoàn thành chuyển đổi!\n")
    
    # Tổng kết
    print("📊 Tổng kết:")
    total_train = len(pd.read_csv(output_dir / 'train.csv'))
    total_val = len(pd.read_csv(output_dir / 'val.csv'))
    total_test = len(pd.read_csv(output_dir / 'test.csv'))
    
    print(f"  Train: {total_train} samples")
    print(f"  Val: {total_val} samples")
    print(f"  Test: {total_test} samples")
    print(f"  Total: {total_train + total_val + total_test} samples")

if __name__ == '__main__':
    convert_dataset()
