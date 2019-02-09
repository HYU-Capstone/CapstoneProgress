import Router from 'koa-router'
import db from '../db'

const router = new Router({ prefix: '/users' })

router.get('/', async ctx => {
  const { rows } = await db.query('SELECT name, id FROM USERS')
  ctx.body = await rows
})

router.post('/', async ctx => {
  const { name, email, phone } = ctx.request.body
  const { rows } = await db.query(
    'INSERT INTO users(name, email, phone) VALUES($1, $2, $3) RETURNING id',
    [name, email, phone]
  )

  ctx.body = await {
    id: rows[0].id,
  }
})

router.get('/batchQuery', async ctx => {
  const { userIds } = ctx.query
  const idList: string[] = JSON.parse(userIds)

  console.log(idList.map(item => `'${item}'`).join(','))
  const { rows } = await db.query(
    'SELECT * FROM users WHERE id = ANY($1::uuid[])',
    [idList]
  )

  ctx.body = await rows
})

router.get('/:userId', async ctx => {
  const { userId } = ctx.params

  const { rows } = await db.query('SELECT * FROM users WHERE id = $1::uuid', [
    userId,
  ])

  ctx.body = await rows[0]
})

router.delete('/:userId', async ctx => {
  const { userId } = ctx.params
  await db.query('DELETE FROM users WHERE id = $1::uuid', [userId])

  ctx.body = await ''
})

router.post('/:userId/changeStatus', async ctx => {
  const { userId } = ctx.params
  const client = await db.startTransaction()
  try {
    await client.query('BEGIN')
    const { rows } = await client.query(
      'SELECT status FROM users WHERE id = $1',
      [userId]
    )
    const status = Boolean(rows[0].status)
    await client.query('UPDATE users SET status = $1 WHERE id = $2', [
      !status,
      userId,
    ])
    await client.query('COMMIT')
    ctx.body = await ''
  } catch (e) {
    await client.query('ROLLBACK')
    ctx.res.statusCode = 500
    ctx.body = await ''
  }
})

export default router
