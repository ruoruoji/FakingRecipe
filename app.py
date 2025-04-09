from flask import Flask, request, jsonify
import torch
from model.FakingRecipe import FakingRecipe_Model

app = Flask(__name__)

# 初始化模型（根据实际路径修改）
model_fakett = FakingRecipe_Model(dataset='fakett')
model_fakett.load_state_dict(torch.load('./provided_ckp/FakingRecipe_fakett'))
model_fakett.eval()

model_fakesv = FakingRecipe_Model(dataset='fakesv') 
model_fakesv.load_state_dict(torch.load('./provided_ckp/FakingRecipe_fakesv'))
model_fakesv.eval()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取请求数据（需要根据实际输入格式调整）
        data = request.json
        video_path = data['video_path']
        
        # 这里需要添加特征提取逻辑（基于当前工程的预处理流程）
        # 示例输入特征，实际需要替换为真实特征提取代码
        input_features = {
            'all_phrase_semantic_fea': torch.randn(1, 512),
            'raw_visual_frames': torch.randn(1, 83, 512),
            # ... 其他必要特征
        }
        
        # 选择模型版本
        model = model_fakett if 'fakett' in video_path else model_fakesv
        
        # 执行预测
        with torch.no_grad():
            output, _, _ = model(**input_features)
            prediction = torch.argmax(output).item()
            
        return jsonify({
            "prediction": int(prediction),
            "confidence": float(torch.softmax(output, dim=1)[0][prediction]),
            "model_used": "fakett" if 'fakett' in video_path else "fakesv"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)