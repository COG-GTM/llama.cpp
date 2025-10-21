import os
import re
import ast
import argparse


def extract_block(text, name):
    pattern = rf'#define\({name}\)\s*(.*?)#end\({name}\)'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Missing block: {name}")
    return match.group(1).strip()


def parse_decls(decls_text):
    decls = {}
    for name, code in re.findall(r'#decl\((.*?)\)\s*(.*?)#enddecl\(\1\)', decls_text, re.DOTALL):
        decls[name.strip()] = code.strip()
    return decls


def replace_placeholders(shader_text, replacements):
    for key, val in replacements.items():
        # Match {{KEY}} literally, where KEY is escaped
        pattern = r'{{\s*' + re.escape(key) + r'\s*}}'
        shader_text = re.sub(pattern, str(val), shader_text)
    return shader_text


def write_shader(shader_name, shader_code, output_dir, outfile):
    if output_dir:
        if not os.path.isdir(output_dir):
            raise ValueError(f"Invalid output directory: {output_dir}")
        wgsl_filename = os.path.join(output_dir, f"{shader_name}.wgsl")
        if not wgsl_filename.startswith(os.path.abspath(output_dir)):
            raise ValueError(f"Path traversal detected: {wgsl_filename}")
        with open(wgsl_filename, "w", encoding="utf-8") as f_out:
            f_out.write(shader_code)
    outfile.write(f'const char* wgsl_{shader_name} = R"({shader_code})";\n\n')


def generate_variants(shader_path, output_dir, outfile):
    if not os.path.isfile(shader_path) or not shader_path.endswith('.wgsl'):
        raise ValueError(f"Invalid shader file: {shader_path}")
    shader_base_name = shader_path.split("/")[-1].split(".")[0]

    with open(shader_path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        variants = ast.literal_eval(extract_block(text, "VARIANTS"))
    except ValueError:
        write_shader(shader_base_name, text, output_dir, outfile)
    else:
        decls_map = parse_decls(extract_block(text, "DECLS"))
        shader_template = extract_block(text, "SHADER")

        for variant in variants:
            decls = variant["DECLS"]
            decls_code = ""
            for key in decls:
                if key not in decls_map:
                    raise ValueError(f"DECLS key '{key}' not found.")
                decls_code += decls_map[key] + "\n\n"

            shader_variant = replace_placeholders(shader_template, variant["REPLS"])
            final_shader = re.sub(r'\bDECLS\b', decls_code, shader_variant)

            output_name = f"{shader_base_name}_" + "_".join([variant["REPLS"]["SRC0_TYPE"], variant["REPLS"]["SRC1_TYPE"]])
            write_shader(output_name, final_shader, output_dir, outfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Invalid input directory: {args.input_dir}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    output_file_path = os.path.abspath(args.output_file)
    with open(output_file_path, "w", encoding="utf-8") as out:
        out.write("// Auto-generated shader embedding\n\n")
        input_dir_abs = os.path.abspath(args.input_dir)
        for fname in sorted(os.listdir(args.input_dir)):
            if fname.endswith(".wgsl"):
                shader_path = os.path.join(input_dir_abs, fname)
                if not shader_path.startswith(input_dir_abs):
                    continue
                generate_variants(shader_path, args.output_dir, out)


if __name__ == "__main__":
    main()
