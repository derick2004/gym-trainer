from flask import Flask, render_template, redirect, url_for, request
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_script', methods=['POST'])
def run_script():
    script_name = request.form['script_name']
    if script_name == 'plank':
        subprocess.Popen(["python", "plank.py"])
    elif script_name == 'pushups':
        subprocess.Popen(["python", "FormFix_Integration.py"])
    elif script_name == 'bicep_curl':
        subprocess.Popen(["python", "bi.py"])
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)