# src/app.py
import os
from flask import Flask, request, jsonify
from src.infer import predict
from werkzeug.utils import secure_filename
import zipfile
import shutil
from src.finetune import finetune_user

UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/infer', methods=['POST'])
def infer_route():
    """
    Send form-data 'audio' field (wav file).
    Optional form-field 'model' = path to model (e.g., models/finetuned_hero.h5)
    """
    if 'audio' not in request.files:
        return jsonify({'error':'No audio file provided'}), 400
    f = request.files['audio']
    filename = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)
    model_path = request.form.get('model', 'models/baseline.h5')
    if not os.path.exists(model_path):
        return jsonify({'error':'Model not found', 'model_path':model_path}), 400
    label, prob = predict(path, model_path=model_path)
    return jsonify({'label':label, 'prob':prob})

@app.route('/finetune', methods=['POST'])
def finetune_route():
    """
    Expects a zip file of a user folder structure:
      user.zip contains:
         play_music/*.wav
         set_alarm/*.wav
         ...
    Form field: 'userzip' (file), 'username' (string)
    """
    if 'userzip' not in request.files:
        return jsonify({'error':'No zip uploaded'}), 400
    username = request.form.get('username', 'unknown')
    z = request.files['userzip']
    fn = secure_filename(z.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
    z.save(save_path)
    extract_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'extracted_{username}')
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    # now call finetune_user on extracted dir
    try:
        finetune_user(extract_dir, base_model_path='models/baseline.h5', user_name=username)
    except Exception as e:
        return jsonify({'error':'finetune failed', 'message':str(e)}), 500
    return jsonify({'status':'fine-tune started/completed for user', 'user':username})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
