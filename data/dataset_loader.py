import pandas as pd
import os
from PIL import Image
import io

class ScienceQALocalLoader:
    def __init__(self, file_path, subset_size=100):
        self.file_path = file_path
        self.subset_size = subset_size
        self.choices_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        try:
            self.df = pd.read_parquet(self.file_path)
        except Exception as e:
            print(f"Lỗi khi đọc file parquet: {e}")
            raise

    def preprocess_for_r3_quant(self):
        """Tiền xử lý dataset cho R3 Quantization"""
        reasoning_col = 'solution' if 'solution' in self.df.columns else 'lecture'
        mask = (self.df[reasoning_col].notnull()) & \
               (self.df[reasoning_col].str.len() > 0) & \
               (self.df['image'].notnull())
        filtered_df = self.df[mask].copy()
        filtered_df = filtered_df.rename(columns={reasoning_col: 'reasoning'})
        
        # Đảm bảo có đủ số lượng mẫu
        if len(filtered_df) < self.subset_size:
            print(f"Warning: Chỉ có {len(filtered_df)} mẫu hợp lệ, nhỏ hơn subset_size={self.subset_size}")
            return filtered_df
        return filtered_df.head(self.subset_size)

    def get_image(self, idx):
        """Lấy image từ dataframe"""
        if 'image' in self.df.columns:
            img_data = self.df.iloc[idx]['image']
            if isinstance(img_data, dict) and 'bytes' in img_data:
                return Image.open(io.BytesIO(img_data['bytes']))
            elif isinstance(img_data, bytes):
                return Image.open(io.BytesIO(img_data))
        return None

    @staticmethod
    def robust_science_qa_matcher(pred, target_letter):
        pred = str(pred).strip().upper()
        patterns = [f"{target_letter}.", f"({target_letter})", f" {target_letter} "]
        if any(p in f" {pred} " for p in patterns) or (len(pred) > 0 and pred[0] == target_letter):
            return 1.0
        return 0.0