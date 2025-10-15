import re
import jax.numpy as jnp
import jax

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