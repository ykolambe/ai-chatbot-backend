import boto3
import json

# Initialize Amazon Bedrock runtime client
bedrock_runtime = boto3.client("bedrock-runtime", region_name="ap-south-1")

def generate_response(prompt: str) -> str:
    """Generates a response using Mistral 7B on Amazon Bedrock."""
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0.7
    })

    try:
        response = bedrock_runtime.invoke_model(
            modelId="mistral.mistral-7b-instruct-v0:2",
            body=body,
            contentType="application/json"
        )

        # Decode and parse the response
        result = json.loads(response["body"].read().decode("utf-8"))
        
        # Extract response text properly
        outputs = result.get("outputs", [])
        if outputs and "text" in outputs[0]:
            return outputs[0]["text"]
        else:
            return "Error: No text found in response."

    except Exception as e:
        return f"Error: {str(e)}"
