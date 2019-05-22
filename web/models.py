class User:
  def __init__(self, user_id, name, email, status, last_checked):
    self.user_id = user_id
    self.name = name
    self.email = email
    self.status = status
    self.last_checked = last_checked

class Attendance:
  def __init__(self, attendance_id, user_id, date, attend_type, user_name):
    self.attendance_id = attendance_id
    self.user_id = user_id
    self.date = date
    self.type = attend_type
    self.user_name = user_name