window.onload = e => {
  const socket = io('http://localhost:5001', {
    forceNew: true,
  })
  socket.on('connect', () => {
    console.log('Connected to SocketIO server')
    socket.on('attend', userId => {
      fetch(`/api/users/${userId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })
        .then(res => {
          return res.json()
        })
        .then(json => {
          if (json.success) {
            const article = document.getElementsByTagName('article')[0]
            article.innerHTML =
              `
            <div class="status-card">
              <h2>${json.data.name} (${json.data.email})</h2>
              <div class="${json.data.status.toLowerCase()}">
                <span class="type">${
                  json.data.status == 'Working' ? '출근' : '퇴근'
                }</span>
                <span class="time">${json.data.last_checked}</span>
              </div>
            </div>
          ` + article.innerHTML
          }
        })
    })
    socket.on('attend-test', msg => {
      const article = document.getElementsByTagName('article')[0]
      article.innerHTML =
        `
        <div class="status-card">
          <h2>${msg} (${'테스트용'})</h2>
          <div class="${'working'}">
            <span class="type">${'출근'}</span>
            <span class="time">${'테스트용'}</span>
          </div>
        </div>
      ` + article.innerHTML
    })
    document.getElementById('test-attendance').addEventListener('click', e => {
      socket.emit('test-request')
    })
  })
}
