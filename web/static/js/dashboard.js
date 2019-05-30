window.onload = (e) => {
  const socket = io()
  socket.on('connect', () => {
    console.log('Connected to SocketIO server')
    socket.on('attend', (msg) => {
      const obj = JSON.parse(msg)
      obj.user_id.forEach((item) => {
        fetch(`/api/users/${item}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          }
        })
        .then(res => {
          return res.json()
        })
        .then(json => {
          if (json.success) {
            const article = document.getElementsByTagName('article')[0]
            article.innerHTML = `
              <div class="status-card">
                <h2>${json.data.name} (${json.data.email})</h2>
                <div class="${json.data.status.toLowerCase()}">
                  <span class="type">${json.data.status == 'Working' ? '출근' : '퇴근'}</span>
                  <span class="time">${json.data.last_checked}</span>
                </div>
              </div>
            ` + article.innerHTML
          }
        })
      })
    })
    document.getElementById('test-attendance').addEventListener('click', (e) => {
      socket.emit('test-request')
    })
  })
}