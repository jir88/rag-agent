import json

from typing import TypedDict, List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from litellm import completion
from langgraph.graph import StateGraph, START, END

from tools import Tool

import phoenix as px
from phoenix.otel import register

from openinference.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry import trace

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# set up Arize Phoenix tracing
# session = px.launch_app()
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = register(
  project_name="researcher-agent", # Default is 'default'
  auto_instrument=True, # Auto-instrument your app based on installed OI dependencies
  endpoint=endpoint,
  batch=True
)
# now that we've set up a provider, grab the actual tracer being used
tracer = trace.get_tracer(__name__)

