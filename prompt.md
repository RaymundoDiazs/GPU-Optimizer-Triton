Voy a confirmar un dato técnico clave antes de entregártela, para que compile a la primera y no pierdas tiempo.Diseñó gramática global acotada para múltiples operadoresDiseñó gramática global acotada para múltiples operadoresConfirmado: XGrammar sigue el formato GBNF y sí soporta repetición acotada {m,n} (de hecho tiene un algoritmo dedicado a comprimirla, así que compila rápido). Esto es justo lo que necesitamos: con límites acotados el modelo puede parar cuando quiera, pero como hay un techo finito, EOS siempre termina siendo alcanzable. Adiós bucle infinito.
Aquí está la gramática completa, lista para compile_grammar():
root ::= module

module ::= imports kernel launcher

# ----- imports (triton + tl obligatorios, torch opcional) -----
imports ::= torch_import triton_import tl_import | triton_import tl_import
torch_import  ::= "import torch\n"
triton_import ::= "import triton\n"
tl_import     ::= "import triton.language as tl\n\n"

# ----- kernel: @triton.jit forzado justo antes del def -----
kernel ::= "@triton.jit\n" "def " ident "(" params "):\n" body

# ----- params: al menos un ": tl.constexpr" GARANTIZADO -----
params      ::= pre_params constexpr_p post_params
pre_params  ::= (plain_p ", "){0,6}
post_params ::= (", " plain_p){0,6}
plain_p     ::= ident
constexpr_p ::= ident ": tl.constexpr"

# ----- body: program_id arriba, store obligatorio, libre alrededor -----
body ::= lead pid_line mid store_line tail
lead ::= free_line{0,4}
mid  ::= free_line{0,40}
tail ::= free_line{0,8}

pid_line   ::= "    " ident " = tl.program_id(" pid_arg ")\n"
store_line ::= "    tl.store(" call_text ")\n"
free_line  ::= "    " line_text "\n"

# ----- launcher / wrapper (obligatorio) -----
launcher      ::= "\ndef " ident "(" call_text "):\n" launcher_body
launcher_body ::= launcher_line{1,20}
launcher_line ::= "    " line_text "\n"

# ----- terminales: sin [^\n], solo clases positivas, todo acotado -----
ident     ::= [a-zA-Z_] [a-zA-Z0-9_]{0,40}
pid_arg   ::= [0-9a-zA-Z_=, ]{1,15}
line_text ::= line_char{1,200}
call_text ::= line_char{1,200}
line_char ::= [a-zA-Z0-9_ \t.,:;=+\-*/%<>()&|\[\]'"]
Cómo resuelve tus dos problemas:

Generación infinita: reemplacé toda la recursión derecha y los +/* por repetición acotada {m,n}. Como cada parte tiene un máximo, llega un punto donde lo único válido es cerrar y emitir EOS. Pero ojo: el modelo puede parar mucho antes (p. ej. apenas termina la línea return del wrapper) — no lo obligo a llenar hasta el tope, solo lo limito.
Shared-prefix / cuelgue del compilador: los choques de prefijo (varias líneas que empiezan con     ) ahora viven dentro de repeticiones acotadas, que XGrammar comprime sin expandirse hasta el infinito. El cuelgue que veías venía solo de los +/recursión sin límite.

Por qué sirve para todos tus operadores: el esqueleto program_id → [líneas libres] → store → [líneas libres] es común a cualquier kernel (add, softmax, matmul, etc.). La parte específica de cada operador va en las free_line, donde line_char permite casi cualquier código (for, if, indexado, aritmética, máscaras). Indentación más profunda (8/12/16 espacios) funciona sola porque el espacio está en line_char.
Dos números que puedes ajustar si algún kernel sale cortado: sube mid de {0,40} (el cuerpo del kernel) o launcher_body de {1,20}. Subirlos no rompe nada, solo da más margen.
Si tu build llega a rechazar algún carácter de line_char o necesitas { } @ para algún operador, dime cuál y lo agrego.