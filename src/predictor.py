import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            # Lấy đường dẫn tuyệt đối của file hiện tại (src/predictor.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Đi lên 1 cấp để về root project (NLP_Transformer)
            project_root = os.path.dirname(current_dir)
            # Tạo đường dẫn đến model
            model_path = os.path.join(project_root, "notebooks", "_models", "phobert-3-labels-final")
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Đang load model từ {model_path} lên {self.device}...")
        
        # Load Tokenizer và Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        
        # Lấy mapping nhãn (0 -> POSITIVE...)
        self.id2label = self.model.config.id2label

    def predict(self, text):
        # Chuẩn bị input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Dự đoán
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Lấy nhãn có xác suất cao nhất
        pred_idx = torch.argmax(probs, dim=-1).item()
        label = self.id2label[pred_idx]
        score = probs[0][pred_idx].item()
        
        return label, score