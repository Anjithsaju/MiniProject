const express = require('express');
const bodyParser = require('body-parser');

const app = express();
const PORT = 5000;

// Middleware
app.use(bodyParser.json());

// Login Route
app.post('/login', (req, res) => {
    const { email, password } = req.body;
    console.log(`Received Login Data -> Email: ${email}, Password: ${password}`);
    
    res.json({ message: "Login data received" });
});

// Start Server
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
