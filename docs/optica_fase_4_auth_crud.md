# Fase 4: Autenticacion y primeros modulos operativos

## Resultado

Se implemento la primera version funcional del CRM interno:

- Login para personal interno.
- Sesion firmada en cookie HTTP-only.
- Proteccion de rutas `/admin`.
- Logout.
- Seed con usuario administrador inicial.
- Dashboard conectado a Prisma.
- CRUD base de clientes.
- CRUD base de productos.
- Inventario inicial con ajustes de stock y movimientos auditables.

## Acceso inicial

Despues de configurar PostgreSQL, ejecutar migraciones y correr el seed:

```text
Email: admin@opticanova.local
Contrasena: Admin123!
```

Estos valores pueden cambiarse con:

```text
ADMIN_EMAIL
ADMIN_PASSWORD
```

## Variables de entorno

Agregar en `.env`:

```text
DATABASE_URL="postgresql://USER:PASSWORD@localhost:5432/optica_crm?schema=public"
AUTH_SECRET="replace-with-a-long-random-secret"
ADMIN_EMAIL="admin@opticanova.local"
ADMIN_PASSWORD="Admin123!"
```

## Comandos para activar base de datos

```bash
cd optica-crm
npm run db:migrate
npm run db:seed
npm run dev
```

## Rutas implementadas

- `/login`
- `/admin`
- `/admin/clientes`
- `/admin/clientes/[id]`
- `/admin/productos`
- `/admin/productos/[id]`
- `/admin/inventario`

## Reglas implementadas

- `/admin` requiere sesion valida.
- La sesion expira despues de 8 horas.
- Clientes se eliminan con baja logica.
- Productos se eliminan con baja logica.
- Productos con inventario requieren SKU.
- Ajustes de inventario guardan movimiento, usuario, producto, sucursal, cantidad y motivo.
- No se permite dejar inventario negativo.

## Pendiente para la siguiente fase

- CRUD de citas.
- CRUD de recetas opticas.
- Flujo de venta completo.
- Asociar recetas a ventas.
- Historial detallado por cliente.
- Roles aplicados a cada accion, no solo sesion general.
