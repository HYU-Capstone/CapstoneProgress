PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE user(
  user_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name VARCHAR(50),
  email VARCHAR(100)
);
CREATE TABLE attendance(
  attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER,
  date DATETIME,
  type VARCHAR(1),
  FOREIGN KEY(user_id) REFERENCES user(user_id)
);
COMMIT;