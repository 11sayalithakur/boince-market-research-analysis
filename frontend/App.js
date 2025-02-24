import React, { useState } from "react";
import axios from "axios";
import "./styles.css";

function App() {
    const [query, setQuery] = useState("");
    const [file1, setFile1] = useState(null);
    const [file2, setFile2] = useState(null);
    const [response, setResponse] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState("");

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");
        setResponse("");

        if (!file1 || !file2) {
            setError("Please upload both PDF files.");
            return;
        }

        if (!query) {
            setError("Please enter a query.");
            return;
        }

        setIsLoading(true);

        const formData = new FormData();
        formData.append("question", query); // Correct key matching FastAPI
        formData.append("files", file1);
        formData.append("files", file2);

        try {
            const res = await axios.post("http://localhost:8000/query/", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            setResponse(res.data.answer);
        } catch (error) {
            console.error(error);
            setError("An error occurred while processing your query.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="App">
            <h1>Market Research Analysis</h1>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Enter your query"
                    required
                />
                <input
                    type="file"
                    accept="application/pdf"
                    onChange={(e) => setFile1(e.target.files[0])}
                    required
                />
                <input
                    type="file"
                    accept="application/pdf"
                    onChange={(e) => setFile2(e.target.files[0])}
                    required
                />
                <button type="submit" disabled={isLoading}>
                    {isLoading ? "Processing..." : "Submit"}
                </button>
            </form>
            {error && <div className="error">{error}</div>}
            <div className="response">
                <h2>Response:</h2>
                {isLoading ? (
                    <div className="loader">Loading...</div>
                ) : (
                    <p>{response}</p>
                )}
            </div>
        </div>
    );
}

export default App;