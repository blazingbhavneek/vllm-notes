# FP16 looping response reproduction

Use the FP16 server command and settings in `serve_fp16.log`, then:

```sh
curl -sS http://127.0.0.1:8001/v1/completions -H 'Content-Type: application/json' --data-binary @fp16_repro_request.json
```

The request is GSM8K index 1. The response from the 250-question run is preserved in `fp16_repro_response.json`: repeated `<think></think>` tokens, finish reason `length`, expected 3. This finding did not raise a Python exception, so there is no failure traceback; complete server and accuracy logs are included.
