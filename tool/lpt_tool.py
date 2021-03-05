import subprocess
import argparse
import os

LPT_PATH=os.getenv("LATEX_PAPERS_ROOT_PATH")

def exec_cmd(cmd):
    cmd_process = subprocess.run(
        cmd
    )
    return str(cmd_process.stdout)

parser = argparse.ArgumentParser(
    description="Latex Papers Tool for automation your work under papers"
)

subparsers = parser.add_subparsers(
    title="commands"
)

# --- UPDATE ---

def update(args):
    exec_cmd(["git", "pull", "https://gitlab.com/iMashtak/latex-papers-template.git", "master"])

update_parser = subparsers.add_parser(
    "update", 
    help="pull all changes from parent Latex Papers Template repository",
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

args = parser.parse_args()
args.func(args)
