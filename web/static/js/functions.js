function fetchUsers() {
  fetch('/api/users', {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json'
    }
  })
  .then(response => {
    return response.json()
  })
  .then(json => {
    if (json.success) {
      const table = document.querySelector('table#users tbody')
      json.data.forEach(item => {
        const tr = document.createElement('tr')
        tr.innerHTML = `
          <td><a href="/users/${item.user_id}">${item.name}</a></td>\n
          <td>${item.email}</td>\n
          <td>${item.status}</td>\n
          <td>${item.last_checked}</td>\n
        `
        table.appendChild(tr)
      })
    } else {
      alert(json.reason)
    }
  })
}

function fetchUser(id) {
  fetch(`/api/users/${id}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json'
    }
  })
  .then(response => {
    return response.json()
  })
  .then(json => {
    if (json.success) {
      const name = document.getElementById('user-name')
      const email = document.getElementById('user-email')
      const status = document.getElementById('user-status')
      const lastChecked = document.getElementById('last-checked')

      name.setAttribute('value', json.data.name)
      email.setAttribute('value', json.data.email)
      status.setAttribute('value', json.data.status)
      lastChecked.setAttribute('value', json.data.last_checked)
    } else {
      alert(json.reason)
    }
  })
}

function updateUser(id) {
  
}

function fetchAttendances() {
  fetch('/api/attendances', {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json'
    }
  })
  .then(response => {
    return response.json()
  })
  .then(json => {
    if (json.success) {
      const table = document.querySelector('table#attendances tbody')
      json.data.forEach(item => {
        const tr = document.createElement('tr')
        tr.innerHTML = `
          <td>${item.user_name}</td>\n
          <td>${item.type}</td>\n
          <td>${item.date}</td>\n
        `
        table.appendChild(tr)
      })
    } else {
      alert(json.reason)
    }
  })
}