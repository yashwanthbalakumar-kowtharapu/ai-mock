    const express = require('express');
    const bodyParser = require('body-parser');
    const bcrypt = require('bcryptjs');
    const jwt = require('jsonwebtoken');
    const mysql = require('mysql2');
    const cors = require('cors');
    const path = require('path');
    require('dotenv').config(); // Load environment variables

    const app = express();
    const PORT = 3000;

    // Middleware
    app.use(cors());
    app.use(bodyParser.json());

    // MySQL Connection
    const db = mysql.createConnection({
        host: 'localhost',
        user: 'root', // Replace with your MySQL username
        password: 'password', // Replace with your MySQL password
        database: 'auth_system'
    });

    db.connect(err => {
        if (err) {
            console.error('Database connection failed:', err);
            process.exit(1);
        }
        console.log('Connected to the MySQL database.');
    });

    // JWT Secret
    const JWT_SECRET = process.env.JWT_SECRET;
    if (!JWT_SECRET) {
        console.error('Missing JWT_SECRET in .env file');
        process.exit(1);
    }

    // Helper Functions
    function generateToken(user) {
        const payload = { userId: user.id, email: user.email };
        return jwt.sign(payload, JWT_SECRET, { expiresIn: '24h' });
    }

    function verifyToken(token) {
        try {
            const decoded = jwt.verify(token, JWT_SECRET);
            return { valid: true, decoded };
        } catch (error) {
            return { valid: false, error: error.message };
        }
    }

    // Routes

    // Serve static files
    app.use(express.static(path.join(__dirname, 'public')));

    app.get('/', (req, res) => {
        res.sendFile(path.join(__dirname, 'public', 'index.html'));
    });

    // Sign-Up Route
    app.post('/signup', async (req, res) => {
        const { username, email, password } = req.body;
        if (!username || !email || !password) {
            return res.status(400).json({ error: 'All fields are required' });
        }

        try {
            const hashedPassword = await bcrypt.hash(password, 10);
            const query = 'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)';
            db.query(query, [username, email, hashedPassword], (err, result) => {
                if (err) {
                    if (err.code === 'ER_DUP_ENTRY') {
                        return res.status(400).json({ error: 'Username or Email already exists' });
                    }
                    throw err;
                }
                res.status(201).json({ message: 'User registered successfully!' });
            });
        } catch (err) {
            res.status(500).json({ error: 'Internal server error' });
        }
    });

    // Login Route
    app.post('/login', async (req, res) => {
        const { email, password } = req.body;

        if (!email || !password) {
            return res.status(400).json({ error: 'Email and password are required' });
        }

        try {
            const query = 'SELECT * FROM users WHERE email = ?';
            db.query(query, [email], async (err, results) => {
                if (err) throw err;

                if (results.length === 0) {
                    return res.status(401).json({ error: 'Invalid email or password' });
                }

                const user = results[0];
                const validPassword = await bcrypt.compare(password, user.password_hash);
                if (!validPassword) {
                    return res.status(401).json({ error: 'Invalid email or password' });
                }

                const token = generateToken(user);
                res.json({ token });
            });
        } catch (err) {
            console.error(err);
            res.status(500).json({ error: 'Internal server error' });
        }
    });

    // Start the server
    app.listen(PORT, () => {
        console.log(`Server is running on http://localhost:${PORT}`);
    });
