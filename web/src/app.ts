import Koa from 'koa'
import koaBody from 'koa-bodyparser'
import koaJson from 'koa-json'
import koaLogger from 'koa-logger'
import { Users } from './routes'

const app = new Koa()

app.use(koaBody({ enableTypes: ['json'] }))
app.use(koaJson())
app.use(koaLogger())

app.use(Users.routes())

app.listen(3000)

console.log('Koa server running on port 3000')
