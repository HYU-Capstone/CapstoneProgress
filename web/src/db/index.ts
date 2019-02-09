import { Pool } from 'pg'
const pool = new Pool({
  user: 'capstone',
  host: 'localhost',
  database: 'capstone',
  password: 'q1w2e3r4',
  port: 5432,
})

export default {
  query: (text: string, params?: any[]) => pool.query(text, params),
  startTransaction: () => pool.connect(),
}
