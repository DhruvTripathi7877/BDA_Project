const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
    username: {
        type: String,
    },
    password: {
        type: String,
    },
    firstName: {
        type: String,
    },
    lastName: {
        type: String,
    },
})

const User= mongoose.model('User', userSchema);

module.exports = {User};