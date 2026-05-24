import argparse
import json

from models.classifier import classify_problem
from parsing.ast_parser import parse_code
from parsing.xgrammar_converter import convert_to_xgrammar
from generation.triton_generator import generate_kernel


def run_pipeline(code: str, validate_gpu: bool = False) -> str:
    problem_type = classify_problem(code)
    ast_representation = parse_code(code)
    grammar = convert_to_xgrammar(ast_representation)
    optimized_kernel = generate_kernel(problem_type, grammar, code=code, ast_repr=ast_representation)

    output = [
        f"Input code: {code}",
        f"Problem type: {problem_type}",
        "AST representation:",
        str(ast_representation),
        "XGrammar representation:",
        grammar,
        "Generated Triton kernel:",
        optimized_kernel,
    ]
    if validate_gpu:
        from validation.gpu_validator import validate_generated_kernel

        validation = validate_generated_kernel(problem_type, optimized_kernel)
        output.extend(
            [
                "GPU validation:",
                json.dumps(validation, indent=2, sort_keys=True),
            ]
        )
    return "\n\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="GPU AI Optimizer prototype: classify code, convert to XGrammar, and generate Triton kernels."
    )
    parser.add_argument(
        "--code",
        default="C = A + B",
        help="Input Python code snippet to optimize. Example: 'C = A + B'",
    )
    parser.add_argument(
        "--output",
        default="results.txt",
        help="Output file to save results. Default: results.txt",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the benchmark suite after generating the requested kernel.",
    )
    parser.add_argument(
        "--validate-gpu",
        action="store_true",
        help="Run the generated Triton wrapper on CUDA and compare it with a PyTorch reference.",
    )
    args = parser.parse_args()

    result = run_pipeline(args.code, validate_gpu=args.validate_gpu)

    # Print to console
    print(result)

    # Save to file
    with open(args.output, "w") as f:
        f.write(result)
    print(f"\nResults saved to {args.output}")

    if args.benchmark:
        from benchmarks.run_benchmarks import run_benchmarks

        rows = run_benchmarks()
        print(f"Benchmark rows written: {len(rows)}")


if __name__ == "__main__":
    main()
