from flask import Blueprint, request, jsonify
import uuid
import MySQLdb
from flask import current_app
import mysql.connector


auth_bp = Blueprint('auth', __name__)

# def get_db_connection():
#     return MySQLdb.connect(
#         host=current_app.config['127.0.0.1'],
#         user=current_app.config['root'],
#         passwd=current_app.config[''],
#         db=current_app.config['Alzwhisper']
#     )
def get_db_connection():
    return mysql.connector.connect(
        host=current_app.config['DB_HOST'],
        user=current_app.config['DB_USER'],
        password=current_app.config['DB_PASSWORD'],
        database=current_app.config['DB_NAME']
    )


@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    db = get_db_connection()
    cursor = db.cursor()

    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    if cursor.fetchone():
        return jsonify({"status": "error", "message": "Email already exists"}), 400

    token = str(uuid.uuid4())
    cursor.execute("INSERT INTO users (name, email, password, uniquetoken) VALUES (%s, %s, %s, %s)",
                   (name, email, password, token))
    db.commit()
    cursor.close()
    db.close()

    return jsonify({"status": "success", "message": "User registered successfully", "token": token})


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    db = get_db_connection()
    cursor = db.cursor()

    cursor.execute("SELECT user_id, name, email, password FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()

    if user and user[3] == password:
        token = str(uuid.uuid4())
        cursor.execute("UPDATE users SET uniquetoken = %s WHERE user_id = %s", (token, user[0]))
        db.commit()

        return jsonify({
            'status': 'success',
            'user': {
                'user_id': user[0],
                'name': user[1],
                'email': user[2],
                'token': token
            }
        }), 200
    else:
        return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401
