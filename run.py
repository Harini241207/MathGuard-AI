from backend.app import app

if __name__ == "__main__":
    print("MathGuard AI running on http://localhost:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)
