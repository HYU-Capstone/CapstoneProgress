CREATE TABLE IF NOT EXISTS users (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4 (),
    name TEXT NOT NULL,
    phone TEXT, 
    email TEXT,
    status BOOLEAN DEFAULT false
);