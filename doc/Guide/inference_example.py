import openai

import httpx
import requests
import time


def wait_for_server_to_come_up(
    url: str = "http://localhost:11434",
    timeout: int = 300,
    time_between_retries: int = 3,
) -> int:
    """This function will sleep until the server responds and returns the status code.
    Or it will throw an exception if the timeout is reached.

    Args:
        url: The server's URL to connect to (e.g. http://localhost:9999).
        timeout: The maximum time in seconds to wait for the server.
        time_between_retries: time in seconds to wait between retries to connect to the server.

    Returns:
        return_code: The status code of the response

    Exception:
        httpx.ConnectError: If the server did not respond within the specified timeout.
    """
    print(f"Waiting for the server at {url} to come up ...")
    start_time = time.time()
    while True:
        try:
            response = httpx.get(url)
            if response.is_success:
                print(f"Server at {url} is up. Status code: {response.status_code}")
            else:
                print(
                    f"Server at {url} responded but might have issues. Status code: {response.status_code}"
                )
            return response.status_code
        except httpx.ConnectError as error:
            if time.time() - start_time > timeout:
                print(
                    f"Timeout reached. The server at {url} is still not up after {timeout}s."
                )
                raise error
            time.sleep(time_between_retries)


def pull_ollama_model(model_name: str, url: str = "http://localhost:11434") -> None:
    """Pulls the specified Ollama model using POST request.

    Args:
        model_name: The name of the model to pull (e.g., "gpt-oss:20b").
    """
    pull_endpoint = f"{url}/api/pull"
    payload = {"model": model_name}
    headers = {"Content-Type": "application/json"}

    response = requests.post(pull_endpoint, json=payload, headers=headers)
    if response.status_code == 200:
        print(f"Successfully pulled model: {model_name}")
    else:
        print(
            f"Failed to pull model: {model_name}. Status code: {response.status_code}, Response: {response.text}"
        )


if __name__ == "__main__":
    url = "http://localhost:11434"
    model = "gpt-oss:20b"

    # wating for the server to come up
    wait_for_server_to_come_up(url=url)

    # pull the model
    pull_ollama_model(model_name=model, url=url)

    # perform inference using OpenAI client
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is deep learning?"},
    ]

    client = openai.OpenAI(base_url=f"{url}/v1", api_key="ollama")
    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        max_tokens=16384,
    )
    print(f"Response from the model:\n{chat_completion.choices[0].message.content}")