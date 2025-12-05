"""
Automatic Data Collection Script
Thu thập data từ public Vietnamese sentiment datasets

Usage:
    python collect_data.py
"""

import pandas as pd
from datasets import load_dataset
import os
from pathlib import Path

class DataCollector:
    def __init__(self, output_dir='../data/collected'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collected_data = []
    
    def collect_uit_vsfc(self):
        """
        Collect UIT-VSFC dataset
        ~16,000 Vietnamese student feedback samples
        """
        print("\n" + "="*60)
        print("Collecting UIT-VSFC Dataset...")
        print("="*60)
        
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset("uitnlp/vietnamese_students_feedback")
            
            # Convert to pandas
            df = dataset['train'].to_pandas()
            
            # Standardize columns
            if 'sentence' in df.columns:
                df = df.rename(columns={'sentence': 'text'})
            if 'label' in df.columns:
                # Map labels to our format
                label_map = {
                    0: 'NEGATIVE',
                    1: 'NEUTRAL', 
                    2: 'POSITIVE'
                }
                df['emotion'] = df['label'].map(label_map)
            
            # Keep only text and emotion columns
            df = df[['text', 'emotion']].copy()
            
            # Save
            output_path = self.output_dir / 'uit_vsfc.csv'
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            self.collected_data.append(df)
            
            print(f"Downloaded: {len(df)} samples")
            print(f"Saved to: {output_path}")
            print(f"\nLabel distribution:")
            print(df['emotion'].value_counts())
            
            return df
            
        except Exception as e:
            print(f"Failed to collect UIT-VSFC: {e}")
            return None
    
    def collect_uit_vsmec(self):
        """
        Collect UIT-VSMEC dataset (your current dataset source)
        """
        print("\n" + "="*60)
        print("Collecting UIT-VSMEC Dataset...")
        print("="*60)
        
        try:
            dataset = load_dataset("SEACrowd/uit_vsmec")
            
            # Combine train and test
            df_train = dataset['train'].to_pandas()
            df_test = dataset['test'].to_pandas() if 'test' in dataset else pd.DataFrame()
            
            df = pd.concat([df_train, df_test], ignore_index=True)
            
            # Standardize
            if 'sentence' in df.columns:
                df = df.rename(columns={'sentence': 'text'})
            
            # Map 6 emotions to 3 sentiments
            emotion_map = {
                'enjoyment': 'POSITIVE',
                'joy': 'POSITIVE',
                'surprise': 'POSITIVE',
                'sadness': 'NEGATIVE',
                'anger': 'NEGATIVE',
                'fear': 'NEGATIVE',
                'disgust': 'NEGATIVE',
                'other': 'NEUTRAL',
                'neutral': 'NEUTRAL'
            }
            
            df['emotion_original'] = df['emotion'].copy()
            df['emotion'] = df['emotion'].str.lower().map(emotion_map)
            
            # Remove unmapped
            df = df[df['emotion'].notna()].copy()
            df = df[['text', 'emotion']]
            
            # Save
            output_path = self.output_dir / 'uit_vsmec.csv'
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            self.collected_data.append(df)
            
            print(f"Downloaded: {len(df)} samples")
            print(f"Saved to: {output_path}")
            print(f"\nLabel distribution:")
            print(df['emotion'].value_counts())
            
            return df
            
        except Exception as e:
            print(f"Failed to collect UIT-VSMEC: {e}")
            return None
    
    def merge_with_current_data(self, current_data_dir='../data/processed_3labels'):
        """
        Merge collected data with current dataset
        """
        print("\n" + "="*60)
        print("Merging with current dataset...")
        print("="*60)
        
        # Load current data
        current_dir = Path(current_data_dir)
        current_dfs = []
        
        for file in ['train.csv', 'val.csv', 'test.csv']:
            file_path = current_dir / file
            if file_path.exists():
                df = pd.read_csv(file_path, encoding='utf-8')
                current_dfs.append(df)
                print(f"  Loaded {file}: {len(df)} samples")
        
        df_current = pd.concat(current_dfs, ignore_index=True)
        print(f"\n  Current total: {len(df_current)} samples")
        
        # Merge all collected data
        all_data = [df_current] + self.collected_data
        df_merged = pd.concat(all_data, ignore_index=True)
        
        print(f"  Before dedup: {len(df_merged)} samples")
        
        # Deduplicate
        df_merged = df_merged.drop_duplicates(subset=['text'], keep='first')
        
        print(f"  After dedup: {len(df_merged)} samples")
        print(f"  New samples added: {len(df_merged) - len(df_current)}")
        
        # Save merged dataset
        output_path = self.output_dir / 'merged_dataset.csv'
        df_merged.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\nMerged dataset saved to: {output_path}")
        print(f"\nFinal label distribution:")
        print(df_merged['emotion'].value_counts())
        
        return df_merged
    
    def split_and_save(self, df, output_dir='../data/processed_3labels_extended'):
        """
        Split merged dataset into train/val/test
        """
        print("\n" + "="*60)
        print("Splitting dataset...")
        print("="*60)
        
        from sklearn.model_selection import train_test_split
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split: 70% train, 10% val, 20% test
        train, temp = train_test_split(df, test_size=0.3, stratify=df['emotion'], random_state=42)
        val, test = train_test_split(temp, test_size=0.67, stratify=temp['emotion'], random_state=42)
        
        print(f"  Train: {len(train)} samples ({len(train)/len(df)*100:.1f}%)")
        print(f"  Val: {len(val)} samples ({len(val)/len(df)*100:.1f}%)")
        print(f"  Test: {len(test)} samples ({len(test)/len(df)*100:.1f}%)")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save
        train.to_csv(output_path / 'train.csv', index=False, encoding='utf-8')
        val.to_csv(output_path / 'val.csv', index=False, encoding='utf-8')
        test.to_csv(output_path / 'test.csv', index=False, encoding='utf-8')
        
        print(f"\nSaved to: {output_path}")
        
        return train, val, test


def main():
    print("="*60)
    print("VIETNAMESE SENTIMENT DATA COLLECTION")
    print("="*60)
    
    collector = DataCollector()
    
    # Collect from public datasets
    collector.collect_uit_vsfc()
    collector.collect_uit_vsmec()
    
    # Merge with current data
    df_merged = collector.merge_with_current_data()
    
    # Split and save
    collector.split_and_save(df_merged)
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETED!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Review collected data in: ../data/collected/")
    print("2. Check merged dataset: ../data/collected/merged_dataset.csv")
    print("3. New train/val/test in: ../data/processed_3labels_extended/")
    print("4. Retrain model with extended dataset")
    print("5. Expected accuracy gain: +4-6%")


if __name__ == '__main__':
    main()
