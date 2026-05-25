import Image from "next/image";
import Link from "next/link";
import {
  ArrowUpRight,
  ChevronRight,
  Menu,
  Search,
  ShoppingBag,
  UserRound,
} from "lucide-react";

const heroImage =
  "https://images.unsplash.com/photo-1612902457652-33aff0a641fa?auto=format&fit=crop&w=1400&q=90";

const offerImage =
  "https://images.unsplash.com/photo-1589782182703-2aaa69037b5b?auto=format&fit=crop&w=900&q=90";

const arrivalImage =
  "https://images.unsplash.com/photo-1606196373155-357259701f44?auto=format&fit=crop&w=900&q=90";

const categories = [
  {
    name: "Armazones",
    count: "128 piezas",
    image:
      "https://images.unsplash.com/photo-1589782182703-2aaa69037b5b?auto=format&fit=crop&w=700&q=85",
  },
  {
    name: "Micas",
    count: "Graduadas",
    image:
      "https://images.unsplash.com/photo-1587304883316-2ce43897de48?auto=format&fit=crop&w=700&q=85",
  },
  {
    name: "Solares",
    count: "Nueva linea",
    image:
      "https://images.unsplash.com/photo-1511499767150-a48a237f0083?auto=format&fit=crop&w=700&q=85",
  },
  {
    name: "Contacto",
    count: "Stock activo",
    image:
      "https://images.unsplash.com/photo-1606196373155-357259701f44?auto=format&fit=crop&w=700&q=85",
  },
];

const navItems = [
  "New arrival",
  "Most pick",
  "Sale",
  "Women",
  "Men",
  "Eye glasses",
  "Sun glass",
  "Contact us",
];

export default function Home() {
  return (
    <main className="min-h-screen bg-[#ffd9cc] p-3 text-black sm:p-5 lg:p-7">
      <section className="mx-auto max-w-[1440px] bg-white px-5 py-6 shadow-[0_20px_80px_rgba(30,20,15,0.08)] sm:px-7 lg:px-8">
        <header className="flex items-center justify-between gap-5">
          <Link href="/" className="flex items-center gap-2" aria-label="Optica Nova">
            <span className="grid size-8 place-items-center rounded-full border-[5px] border-black">
              <span className="size-2 rounded-full bg-black" />
            </span>
            <span className="text-2xl font-semibold tracking-normal">Optica Nova</span>
          </Link>

          <div className="hidden h-10 w-full max-w-[620px] items-center overflow-hidden rounded-full bg-[#eeeeee] text-sm text-zinc-500 lg:flex">
            <span className="px-6">Search</span>
            <span className="ml-auto grid h-full w-12 place-items-center border-l border-zinc-300 text-black">
              <Search size={21} strokeWidth={1.8} />
            </span>
          </div>

          <div className="flex items-center gap-3">
            <Link
              href="/admin"
              className="hidden items-center gap-2 border border-zinc-200 px-4 py-2 text-sm font-medium transition hover:bg-zinc-50 sm:flex"
            >
              CRM
              <ChevronRight size={16} />
            </Link>
            <button
              className="grid size-10 place-items-center rounded-full border border-transparent transition hover:border-zinc-200"
              aria-label="Carrito"
            >
              <ShoppingBag size={22} strokeWidth={1.7} />
            </button>
            <div className="hidden h-10 items-center gap-3 border-l border-zinc-200 pl-4 sm:flex">
              <span className="grid size-8 place-items-center rounded-full bg-[#f6ece5] text-zinc-700">
                <UserRound size={17} />
              </span>
              <span className="text-sm font-medium">David</span>
            </div>
            <button
              className="grid size-10 place-items-center rounded-full bg-zinc-100 lg:hidden"
              aria-label="Menu"
            >
              <Menu size={21} />
            </button>
          </div>
        </header>

        <nav className="mt-7 hidden items-center justify-between gap-5 text-[15px] font-semibold uppercase tracking-normal xl:flex">
          {navItems.map((item) => (
            <a
              key={item}
              href="#catalogo"
              className={item === "Sale" ? "text-[#dd615d]" : "text-zinc-900"}
            >
              {item}
            </a>
          ))}
        </nav>

        <section className="mt-7 grid gap-5 lg:grid-cols-[1.75fr_1fr]">
          <article className="relative min-h-[470px] overflow-hidden rounded-lg bg-[#f8ebe4] lg:min-h-[660px]">
            <Image
              src={heroImage}
              alt="Lentes solares negros sobre tela"
              fill
              className="object-cover"
              priority
              sizes="(min-width: 1024px) 60vw, 100vw"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/72 via-black/20 to-transparent" />
            <div className="absolute bottom-10 left-7 max-w-[440px] text-white sm:left-10">
              <div className="flex items-end gap-1">
                <span className="text-[86px] font-light leading-none sm:text-[112px]">
                  50%
                </span>
                <span className="pb-4 text-3xl font-light uppercase">off</span>
              </div>
              <p className="mt-2 max-w-[390px] text-sm leading-5 text-white/82">
                Lentes seleccionados, armazones de temporada y servicios visuales
                listos para una experiencia mas ligera.
              </p>
              <Link
                href="#catalogo"
                className="mt-7 inline-flex h-12 items-center gap-4 rounded-full bg-white px-8 text-sm font-semibold text-black transition hover:bg-zinc-100"
              >
                Explorar productos
                <ArrowUpRight size={20} />
              </Link>
            </div>
          </article>

          <div className="grid gap-5">
            <article className="relative min-h-[300px] overflow-hidden rounded-lg border border-zinc-100 bg-[#f7f2ee]">
              <Image
                src={offerImage}
                alt="Armazon carey sobre fondo claro"
                fill
                className="object-cover object-left"
                loading="eager"
                sizes="(min-width: 1024px) 34vw, 100vw"
              />
              <div className="absolute inset-0 bg-gradient-to-l from-white/85 via-white/35 to-transparent" />
              <div className="absolute right-7 top-9 text-right">
                <span className="inline-flex border border-black/70 px-5 py-3 text-sm font-semibold uppercase">
                  Para nuevos clientes
                </span>
                <h2 className="mt-7 text-4xl font-semibold uppercase leading-tight">
                  Oferta
                  <br />
                  exclusiva
                </h2>
              </div>
            </article>

            <article className="relative min-h-[300px] overflow-hidden rounded-lg bg-[#eef4f5]">
              <Image
                src={arrivalImage}
                alt="Armazon optico azul sobre mesa clara"
                fill
                className="object-cover"
                loading="eager"
                sizes="(min-width: 1024px) 34vw, 100vw"
              />
              <div className="absolute inset-x-0 bottom-12 flex justify-center">
                <Link
                  href="#catalogo"
                  className="border border-black/70 bg-white/45 px-6 py-3 text-sm font-semibold uppercase backdrop-blur transition hover:bg-white"
                >
                  New arrivals
                </Link>
              </div>
            </article>
          </div>
        </section>

        <section id="catalogo" className="pb-10 pt-16">
          <div className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
            <h1 className="max-w-[900px] text-5xl font-light uppercase leading-none sm:text-6xl lg:text-7xl">
              Browse categories
            </h1>
            <Link
              href="/admin"
              className="inline-flex w-fit items-center gap-2 border border-black px-5 py-3 text-sm font-semibold uppercase transition hover:bg-black hover:text-white"
            >
              Ver CRM
              <ArrowUpRight size={17} />
            </Link>
          </div>

          <div className="mt-10 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
            {categories.map((category) => (
              <article
                key={category.name}
                className="group overflow-hidden rounded-lg bg-[#f7f7f7]"
              >
                <div className="relative h-48 sm:h-56">
                  <Image
                    src={category.image}
                    alt={category.name}
                    fill
                    className="object-cover transition duration-500 group-hover:scale-105"
                    sizes="(min-width: 1024px) 25vw, (min-width: 640px) 50vw, 100vw"
                  />
                </div>
                <div className="flex items-center justify-between px-5 py-4">
                  <div>
                    <h2 className="text-lg font-semibold">{category.name}</h2>
                    <p className="text-sm text-zinc-500">{category.count}</p>
                  </div>
                  <span className="grid size-9 place-items-center rounded-full bg-white text-black shadow-sm">
                    <ArrowUpRight size={18} />
                  </span>
                </div>
              </article>
            ))}
          </div>
        </section>
      </section>
    </main>
  );
}
