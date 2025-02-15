import gradio as gr
import requests
import os

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")


def analyze_text_func(text: str):
    """
    Sends a text input to the /sentiment/analyze_text endpoint.
    """
    try:
        response = requests.post(
            f"{API_URL}/sentiment/analyze_text",
            data={"text": text}
        )
        response.raise_for_status()
        result = response.json()
        prediction = result.get("result", [])
        db_id = result.get("db_id", "N/A")
        if prediction:
            label = prediction[0].get("label", "")
            score = prediction[0].get("score", 0.0)
            output = f"Label: {label}\nScore: {score:.4f}"
        else:
            output = "No prediction returned."
    except Exception as e:
        output = f"Error: {e}"
        db_id = None

    return output, db_id


def analyze_file_func(file_obj, column: str):
    """
    Sends an uploaded file and column name to the /sentiment/analyze_file endpoint.
    """
    if file_obj is None:
        return {"error": "No file uploaded."}
    
    try:
        file_path = file_obj.name
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            data = {"column": column}
            response = requests.post(
                f"{API_URL}/sentiment/analyze_file",
                files=files,
                data=data,
            )
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        result = {"error": str(e)}
    
    return result


def get_result_by_id_func(result_id: int):
    """
    Retrieves a single SentimentResult by ID from the /data/result/{result_id} endpoint.
    """
    try:
        response = requests.get(f"{API_URL}/data/result/{result_id}")
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        result = {"error": str(e)}
    return result


def get_results_by_date_func(date: str):
    """
    Retrieves all SentimentResult records for a given date (YYYY-MM-DD)
    from the /data/results/date endpoint.
    """
    try:
        response = requests.get(f"{API_URL}/data/results/date", params={"date": date})
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        result = {"error": str(e)}
    return result


def validate_result_func(result_id: int, true_result: str):
    """
    Updates the true_result field for a given SentimentResult record.
    Calls the /data/validate/{result_id} endpoint with a query parameter.
    """
    try:
        url = f"{API_URL}/data/validate/{result_id}"
        response = requests.put(url, params={"true_result": true_result})
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        result = {"error": str(e)}
    return result


with gr.Blocks() as demo:
    gr.Markdown("# Sentiment Analysis Frontend")
    
    with gr.Tab("Analyze Text"):
        gr.Markdown("Enter text to analyze its sentiment.")
        text_input = gr.Textbox(label="Text", placeholder="Type your text here...")
        analyze_text_button = gr.Button("Analyze Text")
        text_output = gr.Textbox(label="Prediction")
        db_id_output = gr.Textbox(label="Database Record ID")
        
        analyze_text_button.click(
            fn=analyze_text_func,
            inputs=text_input,
            outputs=[text_output, db_id_output]
        )
    
    with gr.Tab("Analyze File"):
        gr.Markdown("Upload a CSV or Excel file and specify the column name containing the text.")
        file_input = gr.File(label="Upload File", file_types=[".csv", ".xls", ".xlsx"])
        column_input = gr.Textbox(label="Column Name", placeholder="e.g., review_text")
        analyze_file_button = gr.Button("Analyze File")
        file_output = gr.JSON(label="Analysis Results")
        
        analyze_file_button.click(
            fn=analyze_file_func,
            inputs=[file_input, column_input],
            outputs=file_output
        )
    
    with gr.Tab("Retrieve Data"):
        gr.Markdown("Retrieve SentimentResult records by ID or by Date (YYYY-MM-DD).")
        with gr.Row():
            id_input = gr.Number(label="Result ID", value=0, precision=0)
            get_by_id_button = gr.Button("Get Result by ID")
        result_output = gr.JSON(label="Result")
        get_by_id_button.click(
            fn=get_result_by_id_func,
            inputs=id_input,
            outputs=result_output
        )
        
        gr.Markdown("---")
        with gr.Row():
            date_input = gr.Textbox(label="Date (YYYY-MM-DD)", placeholder="2025-02-15")
            get_by_date_button = gr.Button("Get Results by Date")
        results_date_output = gr.JSON(label="Results")
        get_by_date_button.click(
            fn=get_results_by_date_func,
            inputs=date_input,
            outputs=results_date_output
        )
    
    with gr.Tab("Validate Result"):
        gr.Markdown("Validate a SentimentResult record by providing its ID and the true result.")
        validate_id_input = gr.Number(label="Result ID", value=0, precision=0)
        true_result_input = gr.Textbox(label="True Result", placeholder="Enter true result")
        validate_button = gr.Button("Validate Result")
        validate_output = gr.JSON(label="Validation Result")
        
        validate_button.click(
            fn=validate_result_func,
            inputs=[validate_id_input, true_result_input],
            outputs=validate_output
        )

demo.launch()
