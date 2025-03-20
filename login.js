async function login() {
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;

    // Basic validation
    if (!email || !password) {
        alert('Please fill out all fields!');
        return;
    }

    // Email format validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
        alert('Please enter a valid email address!');
        return;
    }

    try {
        // Send login request to the server
        const response = await fetch('http://localhost:3000/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
        });

        const data = await response.json();

        if (response.ok) {
            // Store token and redirect on success
            localStorage.setItem('token', data.token);
            alert('Login successful!');
            window.location.href = 'homepage.html'; // Redirect to homepage
        } else {
            // Show server-side error
            alert(data.error || 'Login failed!');
        }
    } catch (error) {
        console.error('Login Error:', error);
        alert('An error occurred during login. Please try again later.');
    }
}
