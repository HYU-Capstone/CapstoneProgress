import models
import sqlite3

class connect_db:
  def __enter__(self):
    self.conn = sqlite3.connect('users.sqlite')
    self.c = self.conn.cursor()
    return self.c

  def __exit__(self, *exc):
    self.conn.commit()
    self.c.close()
    self.conn.close()

def get_users():
  with connect_db() as cursor:
    users = []
    sql = '''
      SELECT
        A.user_id,
        A.name,
        A.email,
        B.type,
        B.date
      FROM
        user A
      NATURAL JOIN
        (
          SELECT
            MAX(attendance_id) AS id,
            date,
            type,
            user_id AS uid
          FROM
            attendance
          GROUP BY 
            uid
        ) B
      WHERE 
        A.user_id = B.uid 
    '''
    for row in cursor.execute(sql):
      row = list(row)
      row[3] = 'Working' if row[3] == 'I' else 'Leaved'
      users.append(models.User(*row))
    return users

def get_user(user_id):
  with connect_db() as cursor:
    sql = '''
      SELECT
        A.user_id,
        A.name,
        A.email,
        B.type,
        B.date
      FROM
        user A
      NATURAL JOIN
        (
          SELECT
            MAX(attendance_id) AS id,
            date,
            type,
            user_id AS uid
          FROM
            attendance
          GROUP BY 
            uid
        ) B
      WHERE 
        A.user_id = B.uid 
      AND 
        A.user_id = ?
    '''
    cursor.execute(sql, (user_id,))
    row = list(cursor.fetchone())
    row[3] = 'Working' if row[3] == 'I' else 'Leaved'
    return models.User(*row)

def create_user(name, email):
  with connect_db() as cursor:
    sql = 'INSERT INTO users (name, email) VALUES (?, ?)'
    cursor.execute(sql, (name, email, ))
    sql = '''
      SELECT 
        user_id 
      FROM 
        users 
      ORDER BY 
        user_id 
      DESC 
      LIMIT 1
    '''
    cursor.execute(sql)
    return cursor.fetchone()[0]

def update_user(user_id, body):
  with connect_db() as cursor:
    if 'email' not in body.keys() or 'name' not in body.keys():
      raise ValueError('Incorrect POST body - some values are missing')
    sql = 'SELECT * FROM user WHERE user_id = ?'
    cursor.execute(sql, (user_id, ))
    if cursor.fetchone() == None:
      raise ValueError('Invalid ID value')
    sql = 'UPDATE user SET user = ?, email = ? WHERE user_id = ?'
    cursor.execute(sql, (body['user'], body['email'], user_id, ))

def get_attendances():
  with connect_db() as cursor:
    attendances = []
    sql = '''
      SELECT
        A.attendance_id,
        B.user_id,
        A.date,
        A.type,
        B.name
      FROM
        attendance A
      NATURAL JOIN
        user B
      ORDER BY
        attendance_id DESC
      LIMIT
        100
    '''
    for row in cursor.execute(sql):
      row = list(row)
      row[3] = 'Entered' if row[3] == 'I' else 'Leaved'
      attendances.append(models.Attendance(*row))
    return attendances

def get_attendance(att_id):
  with connect_db() as cursor:
    sql = '''
      SELECT
        attendance_id,
        user_id,
        date.
        type
      FROM
        attendance
      WHERE
        attendance_id = ?
    '''
    cursor.execute(sql, (att_id, ))
    row = cursor.fetchone()
    return models.Attendance(*row)
