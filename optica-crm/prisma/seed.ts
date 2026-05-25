import {
  PrismaClient,
  ProductType,
  CategoryType,
  UserStatus,
} from "../src/generated/prisma/client";
import { PrismaPg } from "@prisma/adapter-pg";
import { hashPassword } from "../src/lib/auth/password";

if (!process.env.DATABASE_URL) {
  throw new Error("DATABASE_URL is required to seed the database.");
}

const adapter = new PrismaPg({
  connectionString: process.env.DATABASE_URL,
});

const prisma = new PrismaClient({ adapter });

const roles = [
  ["admin", "Administrador", "Control completo del sistema"],
  ["seller", "Vendedor", "Ventas, clientes y pagos"],
  ["optometrist", "Optometrista", "Agenda y recetas opticas"],
  ["inventory_manager", "Inventario", "Productos, stock y proveedores"],
  ["hr", "Recursos humanos", "Empleados y nomina interna"],
] as const;

const categories = [
  ["Armazones", "armazones", CategoryType.product],
  ["Lentes solares", "lentes-solares", CategoryType.product],
  ["Lentes de contacto", "lentes-contacto", CategoryType.product],
  ["Micas", "micas", CategoryType.product],
  ["Tratamientos", "tratamientos", CategoryType.service],
  ["Accesorios", "accesorios", CategoryType.product],
  ["Servicios", "servicios", CategoryType.service],
] as const;

async function main() {
  const adminEmail = process.env.ADMIN_EMAIL ?? "admin@opticanova.local";
  const adminPassword = process.env.ADMIN_PASSWORD ?? "Admin123!";
  const adminPasswordHash = await hashPassword(adminPassword);

  const organization = await prisma.organization.upsert({
    where: { slug: "optica-nova" },
    update: {},
    create: {
      name: "Optica Nova",
      slug: "optica-nova",
      defaultCurrency: "MXN",
      timezone: "America/Mexico_City",
      phone: "+52 55 0000 0000",
      email: "hola@opticanova.local",
      website: "https://opticanova.local",
    },
  });

  const branch = await prisma.branch.upsert({
    where: {
      organizationId_code: {
        organizationId: organization.id,
        code: "MATRIZ",
      },
    },
    update: {},
    create: {
      organizationId: organization.id,
      name: "Matriz",
      code: "MATRIZ",
      city: "Ciudad de Mexico",
      country: "MX",
      phone: "+52 55 0000 0000",
    },
  });

  const roleRecords = await Promise.all(
    roles.map(([key, name, description]) =>
      prisma.role.upsert({
        where: {
          organizationId_key: {
            organizationId: organization.id,
            key,
          },
        },
        update: { name, description },
        create: {
          organizationId: organization.id,
          key,
          name,
          description,
        },
      }),
    ),
  );

  const admin = await prisma.user.upsert({
    where: {
      organizationId_email: {
        organizationId: organization.id,
        email: adminEmail,
      },
    },
    update: {
      name: "David Admin",
      passwordHash: adminPasswordHash,
      status: UserStatus.active,
    },
    create: {
      organizationId: organization.id,
      email: adminEmail,
      name: "David Admin",
      passwordHash: adminPasswordHash,
      status: UserStatus.active,
    },
  });

  const adminRole = roleRecords.find((role) => role.key === "admin");
  if (adminRole) {
    await prisma.userRole.upsert({
      where: {
        userId_roleId: {
          userId: admin.id,
          roleId: adminRole.id,
        },
      },
      update: {},
      create: {
        userId: admin.id,
        roleId: adminRole.id,
      },
    });
  }

  await prisma.employee.upsert({
    where: { userId: admin.id },
    update: {
      firstName: "David",
      lastName: "Admin",
      jobTitle: "Administrador general",
      branchId: branch.id,
    },
    create: {
      organizationId: organization.id,
      branchId: branch.id,
      userId: admin.id,
      firstName: "David",
      lastName: "Admin",
      email: admin.email,
      jobTitle: "Administrador general",
    },
  });

  const categoryRecords = await Promise.all(
    categories.map(([name, slug, type]) =>
      prisma.category.upsert({
        where: {
          organizationId_slug: {
            organizationId: organization.id,
            slug,
          },
        },
        update: { name, type },
        create: {
          organizationId: organization.id,
          name,
          slug,
          type,
        },
      }),
    ),
  );

  const brand = await prisma.brand.upsert({
    where: {
      organizationId_slug: {
        organizationId: organization.id,
        slug: "nova-atelier",
      },
    },
    update: { name: "Nova Atelier" },
    create: {
      organizationId: organization.id,
      name: "Nova Atelier",
      slug: "nova-atelier",
    },
  });

  const armazones = categoryRecords.find((category) => category.slug === "armazones");
  const solares = categoryRecords.find((category) => category.slug === "lentes-solares");
  const servicios = categoryRecords.find((category) => category.slug === "servicios");

  const productSeeds = [
    {
      categoryId: armazones?.id,
      sku: "ARM-BLK-001",
      name: "Armazon Noir Classic",
      slug: "armazon-noir-classic",
      productType: ProductType.frame,
      salePrice: "1890.00",
      costPrice: "840.00",
      image:
        "https://images.unsplash.com/photo-1574258495973-f010dfbb5371?auto=format&fit=crop&w=900&q=85",
      stock: 18,
    },
    {
      categoryId: solares?.id,
      sku: "SOL-AMBR-001",
      name: "Lente Solar Amber",
      slug: "lente-solar-amber",
      productType: ProductType.sunglasses,
      salePrice: "2290.00",
      costPrice: "980.00",
      image:
        "https://images.unsplash.com/photo-1511499767150-a48a237f0083?auto=format&fit=crop&w=900&q=85",
      stock: 12,
    },
    {
      categoryId: servicios?.id,
      sku: null,
      name: "Examen visual completo",
      slug: "examen-visual-completo",
      productType: ProductType.service,
      salePrice: "450.00",
      costPrice: null,
      image:
        "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?auto=format&fit=crop&w=900&q=85",
      stock: 0,
    },
  ];

  for (const item of productSeeds) {
    if (!item.categoryId) {
      continue;
    }

    const product = await prisma.product.upsert({
      where: {
        organizationId_slug: {
          organizationId: organization.id,
          slug: item.slug,
        },
      },
      update: {
        name: item.name,
        salePrice: item.salePrice,
        costPrice: item.costPrice,
        productType: item.productType,
      },
      create: {
        organizationId: organization.id,
        categoryId: item.categoryId,
        brandId: brand.id,
        sku: item.sku,
        name: item.name,
        slug: item.slug,
        productType: item.productType,
        salePrice: item.salePrice,
        costPrice: item.costPrice,
        trackInventory: item.productType !== ProductType.service,
        reorderPoint: item.productType === ProductType.service ? null : 5,
      },
    });

    await prisma.productImage.deleteMany({ where: { productId: product.id } });
    await prisma.productImage.create({
      data: {
        productId: product.id,
        url: item.image,
        altText: item.name,
        isPrimary: true,
      },
    });

    if (product.trackInventory) {
      await prisma.inventoryStock.upsert({
        where: {
          branchId_productId: {
            branchId: branch.id,
            productId: product.id,
          },
        },
        update: {
          quantityOnHand: item.stock,
          quantityAvailable: item.stock,
        },
        create: {
          organizationId: organization.id,
          branchId: branch.id,
          productId: product.id,
          quantityOnHand: item.stock,
          quantityAvailable: item.stock,
        },
      });
    }
  }
}

main()
  .then(async () => {
    await prisma.$disconnect();
  })
  .catch(async (error) => {
    console.error(error);
    await prisma.$disconnect();
    process.exit(1);
  });
