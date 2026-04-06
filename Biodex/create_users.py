import streamlit_authenticator as stauth
import pickle

# User data
usernames = ['naseem', 'alex']
names = ['Alex Johnson']
passwords = ['password123']

# Correct way to hash in v0.4.2
hasher = stauth.Hasher()
hashed_passwords = [hasher.hash(pw) for pw in passwords]

# Build config
config = {
    'credentials': {
        'usernames': {
            usernames[i]: {
                'name': names[i],
                'password': hashed_passwords[i]
            } for i in range(len(usernames))
        }
    },
    'cookie': {
        'name': 'auth_cookie',
        'key': 'some_random_secret_key',
        'expiry_days': 30
    },
    'preauthorized': {
        'emails': []
    }
}

# Save to pickle
with open('config.pkl', 'wb') as file:
    pickle.dump(config, file)

print("✅ config.pkl with hashed passwords created successfully.")
