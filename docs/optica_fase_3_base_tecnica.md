# Fase 3: Base tecnica del proyecto Next

## Resultado

Se creo una aplicacion Next.js en:

```text
optica-crm/
```

La aplicacion incluye:

- Next.js con App Router.
- TypeScript.
- Tailwind CSS.
- Prisma configurado.
- Esquema inicial de base de datos.
- Seed inicial.
- Estructura base de modulos.
- Ruta publica inicial.
- Ruta administrativa inicial.

## Comandos principales

```bash
cd optica-crm
npm run dev
npm run lint
npm run build
npm run db:generate
npm run db:migrate
npm run db:seed
```

## Base de datos

El archivo `.env.example` contiene la forma esperada de `DATABASE_URL`.

Antes de correr migraciones reales:

1. Crear una base PostgreSQL.
2. Copiar `.env.example` a `.env`.
3. Cambiar usuario, password, host y nombre de base.
4. Ejecutar `npm run db:migrate`.
5. Ejecutar `npm run db:seed`.

## Rutas iniciales

- `/`: sitio publico inspirado en la referencia visual proporcionada.
- `/admin`: primer dashboard del CRM.

## Siguiente fase

La fase 4 debe implementar autenticacion real y conectar los primeros modulos del CRM a la base de datos:

- Login.
- Proteccion de `/admin`.
- CRUD de clientes.
- CRUD de productos.
- Inventario inicial.
