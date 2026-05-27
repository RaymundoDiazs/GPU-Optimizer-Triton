# Fase 1: Alcance inicial para web y CRM de optica

## Objetivo del proyecto

Crear una plataforma web para una optica que combine dos necesidades principales:

1. Una pagina publica para presentar la tienda, servicios, productos y formas de contacto.
2. Un sistema interno tipo CRM/ERP ligero para administrar clientes, inventario, ventas, citas, empleados, nominas y reportes.

El primer objetivo debe ser lanzar un MVP funcional que permita operar lo esencial sin bloquear el crecimiento posterior del sistema.

## Tipo de negocio

La plataforma esta pensada para una optica fisica con posibilidad de venta en linea a futuro.

La tienda puede vender:

- Armazones oftalmicos.
- Lentes solares.
- Lentes de contacto.
- Micas graduadas.
- Tratamientos para lentes.
- Accesorios.
- Servicios de examen visual.
- Reparaciones y ajustes.

Tambien puede gestionar procesos internos como:

- Control de inventario.
- Registro de clientes.
- Historial de recetas.
- Ventas y pagos.
- Citas.
- Empleados.
- Nomina.
- Reportes administrativos.

## Usuarios y roles

### Cliente

Usuario externo que visita la pagina publica.

Permisos principales:

- Ver productos y servicios.
- Solicitar informacion.
- Agendar una cita.
- Enviar datos de contacto.
- Consultar promociones.

En una etapa posterior podria:

- Crear una cuenta.
- Consultar historial de compras.
- Ver recetas registradas.
- Comprar en linea.

### Administrador

Usuario interno con control completo del sistema.

Permisos principales:

- Administrar usuarios internos.
- Ver y editar todos los modulos.
- Gestionar inventario.
- Gestionar ventas.
- Gestionar nomina.
- Consultar reportes.
- Configurar sucursales, categorias, marcas y proveedores.

### Vendedor

Usuario interno enfocado en la operacion comercial diaria.

Permisos principales:

- Registrar clientes.
- Crear ventas.
- Consultar productos disponibles.
- Consultar inventario.
- Registrar pagos.
- Ver historial de compras de un cliente.

Restricciones:

- No puede modificar nomina.
- No puede eliminar productos.
- No puede cambiar configuraciones generales.

### Optometrista

Usuario interno encargado de consultas visuales.

Permisos principales:

- Ver agenda de citas.
- Registrar resultados de examen visual.
- Crear y consultar recetas opticas.
- Consultar historial clinico-optico del cliente.

Restricciones:

- No administra inventario.
- No administra nomina.
- No modifica reportes financieros.

### Encargado de inventario

Usuario interno encargado del stock.

Permisos principales:

- Crear y editar productos.
- Registrar entradas y salidas.
- Ajustar existencias.
- Ver alertas de bajo inventario.
- Gestionar proveedores.

Restricciones:

- No modifica nomina.
- No administra permisos.

### Recursos humanos

Usuario interno encargado de empleados y pagos.

Permisos principales:

- Administrar empleados.
- Registrar asistencias, bonos, descuentos y comisiones.
- Generar calculos preliminares de nomina.
- Consultar historial de pagos a empleados.

Restricciones:

- No gestiona productos ni ventas.
- No modifica configuracion tecnica del sistema.

## Modulos principales

### Sitio publico

Paginas iniciales:

- Inicio.
- Catalogo.
- Detalle de producto.
- Servicios.
- Agendar cita.
- Contacto y ubicacion.
- Promociones.

Objetivo:

Captar clientes, mostrar confianza y facilitar que una persona contacte o visite la optica.

### Autenticacion y permisos

Funciones:

- Inicio de sesion para usuarios internos.
- Roles y permisos.
- Proteccion del panel administrativo.
- Recuperacion de acceso.

Objetivo:

Separar la experiencia publica de la operacion interna.

### Clientes

Funciones:

- Crear clientes.
- Editar datos de contacto.
- Ver historial de compras.
- Ver historial de citas.
- Ver recetas opticas.
- Registrar notas internas.

Datos sugeridos:

- Nombre.
- Telefono.
- Email.
- Fecha de nacimiento.
- Direccion.
- Preferencias.
- Observaciones.

### Recetas opticas

Funciones:

- Registrar graduacion.
- Asociar receta a cliente.
- Asociar receta a venta.
- Consultar historial.

Datos sugeridos:

- Esfera ojo derecho.
- Cilindro ojo derecho.
- Eje ojo derecho.
- Esfera ojo izquierdo.
- Cilindro ojo izquierdo.
- Eje ojo izquierdo.
- Distancia pupilar.
- Adicion.
- Observaciones.
- Fecha de receta.
- Optometrista responsable.

### Productos e inventario

Funciones:

- Crear productos.
- Categorizar productos.
- Registrar marca, modelo y SKU.
- Controlar stock.
- Definir precio de venta y costo.
- Registrar entradas, salidas y ajustes.
- Alertas de bajo inventario.

Categorias iniciales:

- Armazones.
- Lentes solares.
- Lentes de contacto.
- Micas.
- Tratamientos.
- Accesorios.
- Servicios.

### Ventas y pagos

Funciones:

- Crear venta.
- Asociar cliente.
- Agregar productos o servicios.
- Aplicar descuentos.
- Registrar metodo de pago.
- Manejar anticipos y pagos pendientes.
- Emitir ticket o comprobante interno.

Estados sugeridos:

- Cotizacion.
- Apartado.
- Pagado.
- En proceso.
- Listo para entrega.
- Entregado.
- Cancelado.

### Citas

Funciones:

- Crear cita.
- Asignar optometrista.
- Registrar motivo.
- Cambiar estado.
- Asociar cita con cliente.

Estados sugeridos:

- Programada.
- Confirmada.
- Atendida.
- No asistio.
- Cancelada.

### Empleados y nomina

Funciones:

- Crear empleados.
- Asignar puesto.
- Registrar sueldo base.
- Registrar comisiones.
- Registrar bonos y descuentos.
- Generar resumen de pago.

Nota:

La nomina debe iniciar como control administrativo interno. Para calculos fiscales o legales se debe validar la legislacion aplicable antes de automatizar impuestos, retenciones o documentos oficiales.

### Reportes

Reportes iniciales:

- Ventas por periodo.
- Productos mas vendidos.
- Inventario bajo.
- Ventas por vendedor.
- Citas atendidas.
- Clientes recurrentes.
- Margen estimado por producto.
- Resumen de nomina.

## MVP recomendado

La primera version debe incluir lo minimo necesario para operar una optica de forma ordenada.

### Incluido en MVP

- Sitio publico basico.
- Login para personal interno.
- Dashboard administrativo.
- Catalogo de productos.
- Inventario simple.
- Clientes.
- Recetas opticas.
- Ventas.
- Citas.
- Roles basicos: administrador, vendedor y optometrista.

### Fuera del MVP inicial

- Ecommerce completo con pasarela de pago.
- Facturacion fiscal automatica.
- Nomina fiscal avanzada.
- App movil.
- Integracion con WhatsApp Business API.
- Multiples sucursales complejas.
- Programa de lealtad avanzado.

Estos elementos pueden agregarse despues sin bloquear el lanzamiento inicial.

## Flujos principales

### Flujo 1: Cliente agenda una cita

1. El cliente entra al sitio publico.
2. Visita la pagina de servicios o agenda.
3. Ingresa nombre, telefono, fecha deseada y motivo.
4. El sistema crea una cita en estado programada.
5. El personal confirma la cita desde el panel interno.

### Flujo 2: Optometrista registra receta

1. El optometrista abre la cita del cliente.
2. Registra los datos de graduacion.
3. Guarda la receta en el historial del cliente.
4. La receta queda disponible para una venta posterior.

### Flujo 3: Vendedor registra venta

1. El vendedor busca o crea un cliente.
2. Selecciona productos o servicios.
3. Asocia una receta si aplica.
4. Registra descuento, anticipo o pago completo.
5. El sistema descuenta inventario cuando la venta se confirma.
6. La venta queda en estado pagado, en proceso o listo para entrega.

### Flujo 4: Encargado actualiza inventario

1. El encargado abre el modulo de inventario.
2. Busca un producto o crea uno nuevo.
3. Registra una entrada, salida o ajuste.
4. El sistema guarda el movimiento con fecha y usuario responsable.
5. Si el stock queda bajo, se muestra una alerta.

### Flujo 5: Administrador revisa reportes

1. El administrador entra al dashboard.
2. Selecciona rango de fechas.
3. Revisa ventas, inventario bajo, citas y desempeno por vendedor.
4. Exporta o consulta el resumen para tomar decisiones.

### Flujo 6: Recursos humanos calcula nomina interna

1. Recursos humanos abre el modulo de empleados.
2. Selecciona el periodo de pago.
3. Revisa sueldo base, comisiones, bonos y descuentos.
4. Genera un resumen preliminar de pago.
5. Marca el periodo como revisado o pagado.

## Priorizacion sugerida

### Prioridad alta

- Autenticacion.
- Roles basicos.
- Productos.
- Inventario.
- Clientes.
- Recetas.
- Ventas.
- Citas.

### Prioridad media

- Reportes.
- Proveedores.
- Estados de entrega.
- Anticipos y pagos pendientes.
- Comisiones por vendedor.

### Prioridad baja inicial

- Ecommerce.
- Facturacion avanzada.
- Nomina fiscal.
- Programa de puntos.
- Integraciones externas.

## Reglas de negocio iniciales

- Todo producto fisico debe tener SKU.
- Una venta puede existir sin receta, pero una receta siempre debe estar asociada a un cliente.
- El inventario se descuenta cuando una venta se confirma, no cuando solo esta en cotizacion.
- Las citas deben tener cliente, fecha, hora y estado.
- Solo administrador puede eliminar registros criticos.
- Los ajustes de inventario deben guardar motivo y usuario responsable.
- Una receta no debe sobrescribirse; si cambia, se crea una nueva version.
- Una venta cancelada debe restaurar inventario si ya habia sido descontado.

## Riesgos y decisiones pendientes

### Riesgos

- Mezclar demasiado pronto ecommerce, CRM, inventario y nomina puede retrasar el lanzamiento.
- La informacion de recetas opticas requiere cuidado porque es informacion sensible del cliente.
- La nomina puede tener implicaciones legales y fiscales.
- El inventario debe disenar bien sus movimientos para evitar diferencias de stock.

### Decisiones pendientes

- Confirmar si habra una o varias sucursales desde el inicio.
- Confirmar si la tienda vendera en linea desde la primera version.
- Confirmar si se necesita facturacion fiscal.
- Confirmar si el sistema debe integrarse con WhatsApp.
- Confirmar si habra lector de codigo de barras.
- Confirmar moneda y pais de operacion.
- Confirmar si cada producto necesita fotografias multiples.
- Confirmar si las recetas deben poder imprimirse.

## Entregables de la fase 1

- Alcance general definido.
- Roles iniciales definidos.
- Modulos principales definidos.
- Flujos operativos principales definidos.
- MVP recomendado definido.
- Riesgos y decisiones pendientes identificados.

## Siguiente fase

La fase 2 debe convertir este alcance en un modelo de datos.

Entregables esperados de fase 2:

- Entidades principales.
- Relaciones entre entidades.
- Campos por tabla.
- Estados y enums.
- Reglas de integridad.
- Propuesta de stack tecnico para base de datos, ORM y autenticacion.
