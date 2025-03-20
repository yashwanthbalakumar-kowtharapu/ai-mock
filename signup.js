async function signup() {
    const username = document.getElementById('signup-name').value; // Matches 'signup-name'
    const email = document.getElementById('signup-email').value; // Matches 'signup-email'
    const password = document.getElementById('signup-password').value; // Matches 'signup-password'
    const confirmPassword = document.getElementById('signup-confirm-password').value; // Matches 'signup-confirm-password'

    // Basic validation
    if (!username || !email || !password || !confirmPassword) {
        alert('Please fill out all fields!');
        return;
    }

    if (password !== confirmPassword) {
        alert('Passwords do not match!');
        return;
    }

    try {
        // Make the POST request to the server
        const response = await fetch('http://localhost:3000/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password }),
        });

        const result = await response.json();

        if (response.ok) {
            alert('Signup successful! You can now log in.');
            window.location.href = '/index.html'; // Redirect to login page
        } else {
            alert(`Error: ${result.error}`);
        }
    } catch (error) {
        console.error('Error during signup:', error);
        alert('An error occurred during signup. Please try again later.');
    }
}
