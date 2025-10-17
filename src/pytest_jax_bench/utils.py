import os
import re
import jax.numpy as jnp
import jax
from difflib import SequenceMatcher, unified_diff

# ------------------------------------------------------------------------------------------------ #
#                             Detecting folded constants in HLO graphs                             #
# ------------------------------------------------------------------------------------------------ #

def token_to_jnp_dtype(tok: str):
    """Map MLIR/StableHLO tokens to JAX/NumPy dtypes."""
    # booleans
    if tok == "i1":
        return jnp.bool_

    # complex
    if tok.startswith("complex<") and tok.endswith(">"):
        inner = tok[len("complex<"):-1]
        return {"f32": jnp.complex64, "f64": jnp.complex128}[inner]

    # integers: si*/ui*/i*
    m = re.fullmatch(r'(si|ui|i)(\d+)', tok)
    if m:
        kind, bits = m.groups()
        bits = int(bits)
        if kind == "ui":
            return jnp.dtype(f"uint{bits}")
        elif kind in ("si", "i"):
            return jnp.dtype(f"int{bits}")

    # fp8 & microscaling families
    FP_MAP = {
        "f8E3M4": jnp.float8_e3m4,
        "f8E4M3": jnp.float8_e4m3,
        "f8E4M3FN": jnp.float8_e4m3fn,
        "f8E4M3FNUZ": jnp.float8_e4m3fnuz,
        "f8E4M3B11FNUZ": jnp.float8_e4m3b11fnuz,
        "f8E5M2": jnp.float8_e5m2,
        "f8E5M2FNUZ": jnp.float8_e5m2fnuz,
        "f8E8M0FNU": jnp.float8_e8m0fnu,
        "f4E2M1FN": jnp.float4_e2m1fn,
        # "f6E2M3FN": jnp.float6_e2m3fn,
        # "f6E3M2FN": jnp.float6_e3m2fn,
    }
    if tok in FP_MAP:
        return FP_MAP[tok]

    # standard floats + bf16 + tf32
    if tok in {"bf16", "f16", "f32", "f64"}:
        return {"bf16": jnp.bfloat16, "f16": jnp.float16,
                "f32": jnp.float32, "f64": jnp.float64}[tok]
    if tok == "tf32":
        return jnp.float32  # StableHLO 'tf32' ⇒ closest array dtype is float32
    
    print(f"Warning: unknown dtype token {tok}. I'll assume float32 instead")
    return jnp.float32

def shape_dtype_to_struct(spec: str) -> jax.ShapeDtypeStruct:
    """
    Convert a captured 'shapexdtype' spec from tensor<...> into ShapeDtypeStruct.
    Examples:
      '131584x2xf32' -> shape=(131584, 2), dtype=float32
      'f32'          -> shape=(), dtype=float32 (scalar tensor)
    """
    m2 = re.match(
        r'^(?:(\d+(?:x\d+)*)x)?('
        r'complex<[^<>]+>|'
        r'i1|'
        r'(?:si|ui)(?:2|4|8|16|32|64)|'      # si*/ui*
        r'i(?:8|16|32|64)|'                  # legacy i*
        r'bf16|f16|f32|f64|tf32|'
        r'f(?:4E2M1FN|6E2M3FN|6E3M2FN|'
        r'8E3M4|8E4M3(?:B11FNUZ|FNUZ|FN)?|'
        r'8E5M2(?:FNUZ)?|8E8M0FNU)'
        r')$',
        spec
    )
    if not m2:
        raise ValueError(f"Unparsable tensor spec: {spec}")
    dims = tuple(map(int, m2.group(1).split('x'))) if m2.group(1) else ()
    dtype_tok = m2.group(2)
    return jax.ShapeDtypeStruct(shape=dims, dtype=token_to_jnp_dtype(dtype_tok))

PATTERN = re.compile(
    r'(?m)^\s*%cst(?:_\d+)?\s*=\s*stablehlo\.constant\b[^\n]*?:\s*tensor<((?:[^<>]|<[^<>]*>)+)>'
)
def detect_folded_constants(low):
    """Return a list of ShapeDtypeStruct for all %cst tensor types of a lowered jax function
    
    example: detect_folded_constants(jax.jit(f).lower(x))
    """
    shlo = low.compiler_ir(dialect="stablehlo")
    txt = shlo.operation.get_asm(large_elements_limit=16)
    return [shape_dtype_to_struct(m.group(1)) for m in PATTERN.finditer(txt)]

def folded_constants_bytes(low):
    """Return the total size in bytes of all folded constants in a lowered jax function
    
    example: folded_constants_bytes(jax.jit(f).lower(x))
    """
    consts = detect_folded_constants(low)
    return sum(c.size * c.dtype.itemsize for c in consts)


# ------------------------------------------------------------------------------------------------ #
#                                 Helpers for generating SVG garphs                                #
# ------------------------------------------------------------------------------------------------ #

def hlo_to_svg_text(hlo_text: str):
    from jaxlib import xla_client as xc
    from graphviz import Source

    # Parse HLO text -> XLA HloModule, then emit DOT and render to SVG
    mod = xc._xla.hlo_module_from_text(hlo_text)
    dot = xc._xla.hlo_module_to_dot_graph(mod)
    svg_bytes = Source(dot).pipe(format="svg")

    if isinstance(svg_bytes, bytes):
        svg_text = svg_bytes.decode("utf-8")
    else:
        svg_text = str(svg_bytes)

    return svg_text

# ------------------------ Detecting whether svg files describe same graph ----------------------- #

_PI_XML    = re.compile(r'<\?xml[\s\S]*?\?>', re.S)                 # XML prolog
_PI_STYLE  = re.compile(r'<\?xml-stylesheet[\s\S]*?\?>', re.S)      # (multi-line) stylesheet PI
_DOCTYPE   = re.compile(r'<!DOCTYPE[\s\S]*?>', re.S)
COMMENTS   = re.compile(r'<!--[\s\S]*?-->', re.S)
TITLES     = re.compile(r'<title>[\s\S]*?</title>', re.S)           # kill ALL titles
NUMBERS    = re.compile(r'\d+(?:\.\d+)?')
WS_MULTI   = re.compile(r'\s+')

def _light_norm(svg: str, zero_numbers: bool = True) -> str:
    s = _PI_XML.sub('', svg)
    s = _PI_STYLE.sub('', s)    # this is the big fix (handles multiline)
    s = _DOCTYPE.sub('', s)
    s = COMMENTS.sub('', s)
    s = TITLES.sub('', s)       # strip all <title>…</title>
    # optional: kill all numbers (IDs, coords, counters)
    if zero_numbers:
        s = NUMBERS.sub('0', s)
    # collapse whitespace
    s = WS_MULTI.sub(' ', s).strip()
    return s

def _orderless_signature(svg: str) -> str:
    s = _light_norm(svg, zero_numbers=True)
    # split at tag boundaries: "...><..."
    parts = re.split(r'(?<=>)\s*(?=<)', s)
    parts = [p.strip() for p in parts if p.strip()]
    parts.sort()
    return '\n'.join(parts)

def svgs_close(svg_a: str, svg_b: str, threshold: float = 0.995):
    sa = _orderless_signature(svg_a)
    sb = _orderless_signature(svg_b)
    if sa == sb:
        return True, 1.0
    sim = SequenceMatcher(None, sa, sb).ratio()
    return sim >= threshold, sim

def save_graph_svg(f_comp, filepath : str, only_if_different : str | None = None):
    hlo_text = f_comp.as_text()
    svg = hlo_to_svg_text(hlo_text)

    if only_if_different is not None:
        if os.path.exists(only_if_different):
            with open(only_if_different, "r", encoding="utf-8") as f:
                existing_svg = f.read()
            close, sim = svgs_close(svg, existing_svg)
            if close:
                return
            else:
                print(f"Existing SVG differs (similarity {sim:.3f}); Creating new at {filepath}")

    with open(filepath, "w") as f:
        f.write(svg)