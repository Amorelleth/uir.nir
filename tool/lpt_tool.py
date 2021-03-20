import subprocess
import argparse
import os

LPT_PATH=os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/../")

def exec_cmd(cmd):
    cmd_process = subprocess.run(
        cmd,
        stderr=subprocess.STDOUT
    )
    return cmd_process

def get_version():
    with open(f"{LPT_PATH}/.version", "r") as v:
        return v.read()

parser = argparse.ArgumentParser(
    description=f"Latex Papers Tool v{get_version()}"
)

subparsers = parser.add_subparsers(
    title="commands"
)

# --- UPDATE ---

def update(args):
    exec_cmd(["git", "pull", "https://gitlab.com/iMashtak/latex-papers-template.git", "master"])

update_parser = subparsers.add_parser(
    "update", 
    help="pull all changes from root Latex Papers Template repository",
)
update_parser.set_defaults(func=update)

# --- CREATE ---

def create(args):
    src_path = f"{LPT_PATH}/templates/{args.template_name}"
    temp_path = f"{LPT_PATH}/src/"
    dest_path = f"{LPT_PATH}/src/{args.template_name}.{args.paper_name}"
    exec_cmd(["rsync", "-av", f"{src_path}", temp_path, "--exclude", "out"])
    exec_cmd(["mv", f"{temp_path}/{args.template_name}", dest_path])

create_parser = subparsers.add_parser(
    "create", 
    help="create paper from template"
)
create_parser.add_argument(
    "template_name", 
    help="name of template"
)
create_parser.add_argument(
    "paper_name",
    help="name of dir created at /src"
)
create_parser.set_defaults(func=create)

# --- LIST TEMPLATES ---

def list_templates(args):
    files = os.listdir(f"{LPT_PATH}/templates")
    for file in files:
        if os.path.isdir(f"{LPT_PATH}/templates/{file}"):
            print(file)

list_templates_parser = subparsers.add_parser(
    "ls",
    help="list all available templates with description"
)
list_templates_parser.set_defaults(func=list_templates)

# --- CHECK PROTECTION ---

# def check_protection(args):
#     print("Checking protection")

# check_protection_parser = subparsers.add_parser(
#     "check",
#     help="check if there any changes in protected files"
# )
# check_protection_parser.set_defaults(func=check_protection)

# --- BUILD ---

def build(args):
    doc_abs_path = args.document
    doc_parts = doc_abs_path.split('/')
    delim_index = -1
    for i, item in enumerate(doc_parts):
        if item == "latex-papers-template":
            delim_index = i
    doc = '/'.join(doc_parts[delim_index+1:])
    if not doc.endswith(".tex"):
        doc += ".tex"
    dirname = '/'.join(doc_parts[delim_index+1:-1])
    image = "deralusws/latex-papers-template-image:1.0"
    exec_cmd([
        "docker", "run", "--rm", "-i", "-v", f"{LPT_PATH}:/data", image,
        "latexmk", "-synctex=1", "-interaction=nonstopmode", "-file-line-error", "-xelatex", f"-outdir={dirname}/out", f"{doc}"
    ])

build_parser = subparsers.add_parser(
    "build",
    help="builds latex document"
)
build_parser.add_argument(
    "document",
    help="path to root latex document file"
)
build_parser.set_defaults(func=build)

args = parser.parse_args()
try:
    args.func(args)
except Exception as e:
    print(e)
    parser.print_help()
