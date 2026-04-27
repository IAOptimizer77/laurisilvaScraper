import asyncio
import base64
import hashlib
import json as _json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import phpserialize
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from playwright.async_api import async_playwright, Page
from playwright_stealth import stealth_async
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, UpdateStatus

# ============================================================
# Configuración
# ============================================================

QDRANT_URL      = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536
BATCH_SIZE      = 50
PAGE_TIMEOUT    = 30_000  # ms — timeout por página en Playwright
SCRAPE_DELAY    = float(os.getenv("SCRAPE_DELAY", "1.0"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("scraper")

# ============================================================
# Modelo de datos
# ============================================================

@dataclass
class Producto:
    nombre: str
    precio: str
    marca: str
    descripcion: str
    categoria: str
    sku: str
    url: str
    tienda: str
    formato: str = ""

    def texto_para_embedding(self) -> str:
        return f"{self.nombre} {self.marca} {self.categoria} {self.formato} {self.descripcion}"

    def uid(self) -> int:
        return int(hashlib.md5(self.url.encode()).hexdigest(), 16) % (2**63)

    def content_hash(self) -> str:
        raw = f"{self.nombre}|{self.precio}|{self.marca}|{self.descripcion}"
        return hashlib.md5(raw.encode()).hexdigest()

    def precio_float(self) -> float:
        try:
            return float(self.precio.replace("€", "").replace(",", ".").strip())
        except Exception:
            return 0.0

    def payload(self) -> dict:
        return {
            "NombreProducto":      self.nombre,
            "Precio":              self.precio_float(),
            "Marca":               self.marca,
            "InformacionProducto": self.descripcion,
            "Categoria":           self.categoria,
            "Formato":             self.formato,
            "SKU":                 self.sku,
            "URL":                 self.url,
            "Tienda":              self.tienda,
            "content_hash":        self.content_hash(),
        }

# ============================================================
# Sitemap — requests (sin Playwright, ya funciona)
# ============================================================

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
SITEMAP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "Accept": "application/xml,text/xml,*/*",
    "Accept-Language": "es-ES,es;q=0.9",
}


def get_urls_from_sitemap(sitemap_url: str) -> list[str]:
    try:
        r = requests.get(sitemap_url, headers=SITEMAP_HEADERS, timeout=15)
        r.raise_for_status()
        content = r.content
        if content[:1] != b"<":
            logger.error(f"Sitemap {sitemap_url} devolvió no-XML. Primeros 200 bytes: {content[:200]}")
            return []
        root = ET.fromstring(content)
        urls = [loc.text.strip() for loc in root.findall(".//sm:loc", SITEMAP_NS) if loc.text]
        logger.info(f"Sitemap {sitemap_url} → {len(urls)} URLs")
        return urls
    except Exception as e:
        logger.error(f"Error leyendo sitemap {sitemap_url}: {e}")
        return []


def get_product_urls_ventana_natural() -> list[str]:
    import re
    all_urls = get_urls_from_sitemap("https://laventananatural.com/1_es_0_sitemap.xml")
    pattern = re.compile(r"laventananatural\.com/[^/]+/\d+-[^/]+\.html$")
    products = [u for u in all_urls if pattern.search(u)]
    logger.info(f"La Ventana Natural — {len(products)} productos")
    return products


# ============================================================
# LauriSilvaBio — scraping por páginas de categoría (Arminet)
# ============================================================
# Cada página de categoría devuelve 20 productos con datos completos
# embebidos en widget `lista_deseos-XXXXX` data-configuration (base64
# → PHP serialize → objeto Articulo). Funciona desde IP datacenter,
# Cloudflare no bloquea categorías. Ver auditoria/INVESTIGACION_LAURISILVA_SCRAPER.md

LAURISILVA_CATEGORIAS = [
    "herbolario",
    "dieta",
    "cosmetica-natural",
    "azucar%2c-edulcorantes-y-mieles",
    "infusiones-y-tes",
    "cafe-y-cacaos",
    "legumbres-y-pastas",
    "mermeladas%2c-cremas-y-pates",
    "otros",
    "reposteria%2c-galletas-y-panes",
    "superalimentos%2c-raw-polvo",
    "semillas-y-cereales",
    "sin-gluten",
    "sales%2c-aceites%2c-vinagres-y-salsas",
    "aromaterapia",
    "infantil-y-mama",
    "vivir-sin-plastico",
    "ofertas",
    "chocolates",
    "harinas",
    "frutos-secos",
    "tortitas-y-barritas",
]

LAURISILVA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "es-ES,es;q=0.9",
}

LISTA_DESEOS_RE = re.compile(r"^lista_deseos-\d+$")


def _php_object_hook(name, d):
    return phpserialize.phpobject(name, d)


def _to_dict(obj):
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj if isinstance(obj, dict) else {}


def _php_str(obj, key: str, default: str = "") -> str:
    """Lee campo string de Articulo deserializado (claves bytes)."""
    val = obj.get(key.encode()) if isinstance(obj, dict) else None
    if val is None:
        return default
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", errors="replace")
        except Exception:
            return default
    return str(val)


def _php_num(obj, key: str, default: float = 0.0) -> float:
    val = obj.get(key.encode()) if isinstance(obj, dict) else None
    try:
        return float(val) if val is not None else default
    except Exception:
        return default


def parse_categoria_laurisilva(html: str) -> list[Producto]:
    """Extrae productos de una página de categoría laurisilva."""
    soup = BeautifulSoup(html, "lxml")
    productos: list[Producto] = []

    for div in soup.find_all("div", attrs={"data-update": LISTA_DESEOS_RE}):
        config_b64 = div.get("data-configuration", "")
        if not config_b64:
            continue
        try:
            raw = base64.b64decode(config_b64)
            outer = phpserialize.loads(raw, decode_strings=False, object_hook=_php_object_hook)
            opciones = outer.get(b"opciones", {}) if isinstance(outer, dict) else {}
            articulo = opciones.get(b"producto") if isinstance(opciones, dict) else None
            if articulo is None:
                continue
            art = _to_dict(articulo)
            if not art:
                continue

            nombre = _php_str(art, "nombre")
            url = _php_str(art, "url")
            if not nombre or not url:
                continue

            ean = _php_str(art, "ean")
            sku = ean or _php_str(art, "referencia") or _php_str(art, "codigo")
            precio_val = _php_num(art, "pvp_final") or _php_num(art, "pvp")
            marca = _php_str(art, "autor")
            obs_html = _php_str(art, "observaciones")
            descripcion = BeautifulSoup(obs_html, "lxml").get_text(" ", strip=True)[:1000] if obs_html else ""
            categoria = ""
            temas = art.get(b"temas")
            if isinstance(temas, dict) and temas:
                first_tema = next(iter(temas.values()), None)
                if first_tema is not None:
                    categoria = _php_str(_to_dict(first_tema), "nombre")

            productos.append(Producto(
                nombre=nombre,
                precio=f"{precio_val:.2f}",
                marca=marca,
                descripcion=descripcion,
                categoria=categoria,
                sku=str(sku),
                url=url,
                tienda="LauriSilvaBio",
            ))
        except Exception as e:
            logger.debug(f"Error parsing widget: {e}")
            continue

    return productos


def scrape_laurisilva_categorias(qdrant: QdrantClient, openai_client: OpenAI):
    """Scrape laurisilva paginando categorías. Sin Playwright."""
    collection = "laurisilva_productos"
    ensure_collection(qdrant, collection)
    existing_hashes = fetch_existing_hashes(qdrant, collection)

    seen_uids: set[int] = set()
    batch: list[Producto] = []
    ok = skipped = err = pages = 0

    for cat in LAURISILVA_CATEGORIAS:
        page = 0
        empty_streak = 0
        while True:
            url = f"https://www.laurisilvabio.com/{cat}" if page == 0 else f"https://www.laurisilvabio.com/{cat}-p{page}"
            try:
                r = requests.get(url, headers=LAURISILVA_HEADERS, timeout=30)
                if r.status_code != 200:
                    logger.warning(f"[{cat}] page {page} → HTTP {r.status_code}")
                    break
                productos = parse_categoria_laurisilva(r.text)
            except Exception as e:
                logger.warning(f"[{cat}] page {page} error: {e}")
                err += 1
                break

            pages += 1
            if not productos:
                empty_streak += 1
                if empty_streak >= 1:
                    break
            else:
                empty_streak = 0

            new_in_page = 0
            for p in productos:
                uid = p.uid()
                if uid in seen_uids:
                    continue
                seen_uids.add(uid)
                new_in_page += 1
                if existing_hashes.get(uid) == p.content_hash():
                    skipped += 1
                else:
                    batch.append(p)
                    ok += 1

            if len(batch) >= BATCH_SIZE:
                upsert_batch(qdrant, collection, batch, openai_client)
                logger.info(f"[laurisilva] cat={cat} page={page} upserted={ok} skipped={skipped} unique={len(seen_uids)}")
                batch = []

            if new_in_page == 0 and page > 0:
                break

            page += 1
            time.sleep(SCRAPE_DELAY)

    if batch:
        upsert_batch(qdrant, collection, batch, openai_client)

    logger.info(f"[laurisilva] COMPLETO — {ok} actualizados, {skipped} sin cambios, {err} errores, {pages} páginas, {len(seen_uids)} productos únicos")


def get_product_urls_laurisilva(qdrant_client: QdrantClient = None) -> list[str]:
    """
    Returns product URLs for laurisilva scraping.
    Primary source: existing Qdrant collection (avoids Cloudflare-blocked sitemaps).
    Fallback: sitemap via requests (works only from non-datacenter IPs).
    """
    if qdrant_client is not None:
        urls = _get_urls_from_qdrant(qdrant_client, "laurisilva_productos")
        if urls:
            return urls

    # Fallback: sitemaps (blocked from Easypanel/datacenter IPs)
    sitemaps = [
        "https://www.laurisilvabio.com/product_sitemap_0.xml",
        "https://www.laurisilvabio.com/product_sitemap_1000.xml",
        "https://www.laurisilvabio.com/product_sitemap_2000.xml",
        "https://www.laurisilvabio.com/product_sitemap_3000.xml",
    ]
    all_urls = []
    for sm in sitemaps:
        all_urls.extend(get_urls_from_sitemap(sm))
    logger.info(f"LauriSilvaBio — {len(all_urls)} productos (desde sitemap)")
    return all_urls


def _get_urls_from_qdrant(client: QdrantClient, collection: str) -> list[str]:
    """Scroll all points and extract URL from payload."""
    urls = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=offset,
            with_payload=["URL"],
            with_vectors=False,
        )
        for p in points:
            url = p.payload.get("URL") or p.payload.get("metadata", {}).get("URL")
            if url:
                urls.append(url)
        if offset is None:
            break
    logger.info(f"[{collection}] {len(urls)} URLs extraídas de Qdrant")
    return urls

# ============================================================
# Parsers HTML (reciben el HTML ya renderizado por Playwright)
# ============================================================

def parse_ventana_natural(html: str, url: str) -> Optional[Producto]:
    soup = BeautifulSoup(html, "lxml")

    nombre_el = (
        soup.select_one("h1.product_name, h1.page-heading, h1[itemprop='name']") or
        soup.select_one("h1")
    )
    nombre = nombre_el.get_text(strip=True) if nombre_el else ""
    if not nombre:
        return None

    precio_el = (
        soup.select_one("span[itemprop='price'], span.current-price-value, .product-price") or
        soup.select_one(".price")
    )
    precio = precio_el.get_text(strip=True) if precio_el else ""

    marca_el = (
        soup.select_one("span[itemprop='brand'], .manufacturer-name, a.manufacturer") or
        soup.select_one("[itemprop='brand']")
    )
    marca = marca_el.get_text(strip=True) if marca_el else ""

    desc_el = (
        soup.select_one("div[itemprop='description'], #short_description_block, .product-description") or
        soup.select_one(".product_description")
    )
    descripcion = desc_el.get_text(separator=" ", strip=True)[:1000] if desc_el else ""

    breadcrumbs = soup.select("nav[aria-label='breadcrumb'] li, ol.breadcrumb li, .breadcrumb li")
    categoria = " > ".join(b.get_text(strip=True) for b in breadcrumbs[1:-1]) if len(breadcrumbs) > 2 else ""

    sku_el = soup.select_one("[itemprop='sku'], .product-reference span")
    sku = sku_el.get_text(strip=True) if sku_el else url.split("-")[-1].replace(".html", "")

    return Producto(nombre=nombre, precio=precio, marca=marca, descripcion=descripcion,
                    categoria=categoria, sku=sku, url=url, tienda="La Ventana Natural")


def parse_laurisilva(html: str, url: str) -> Optional[Producto]:
    soup = BeautifulSoup(html, "lxml")

    # JSON-LD primero (WooCommerce/Shopify lo incluyen)
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = _json.loads(script.string or "")
            if isinstance(data, list):
                data = next((d for d in data if d.get("@type") == "Product"), None)
            if data and data.get("@type") == "Product":
                offers = data.get("offers", {})
                if isinstance(offers, list):
                    offers = offers[0] if offers else {}
                precio = f"{offers.get('price', '')} {offers.get('priceCurrency', '')}".strip()
                marca_data = data.get("brand", {})
                marca = marca_data.get("name", "") if isinstance(marca_data, dict) else str(marca_data)
                return Producto(
                    nombre=data.get("name", ""),
                    precio=precio,
                    marca=marca,
                    descripcion=BeautifulSoup(data.get("description", ""), "lxml").get_text()[:1000],
                    categoria=data.get("category", ""),
                    sku=str(data.get("sku", "")),
                    url=url,
                    tienda="LauriSilvaBio",
                )
        except Exception:
            continue

    # Fallback HTML
    nombre_el = soup.select_one("h1")
    nombre = nombre_el.get_text(strip=True) if nombre_el else ""
    if not nombre:
        return None

    precio_el = soup.select_one(".price, .product-price, [itemprop='price']")
    marca_el  = soup.select_one(".brand, [itemprop='brand'], .manufacturer")
    desc_el   = soup.select_one(".description, [itemprop='description'], .product-description")

    return Producto(
        nombre=nombre,
        precio=precio_el.get_text(strip=True) if precio_el else "",
        marca=marca_el.get_text(strip=True) if marca_el else "",
        descripcion=desc_el.get_text(separator=" ", strip=True)[:1000] if desc_el else "",
        categoria="", sku="", url=url, tienda="LauriSilvaBio",
    )

# ============================================================
# Qdrant
# ============================================================

def fetch_existing_hashes(client: QdrantClient, collection: str) -> dict[int, str]:
    """Carga {point_id: content_hash} de todos los puntos existentes en una pasada."""
    hashes: dict[int, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=offset,
            with_payload=["content_hash"],
            with_vectors=False,
        )
        for p in points:
            h = p.payload.get("content_hash")
            if h:
                hashes[p.id] = h
        if offset is None:
            break
    logger.info(f"[{collection}] {len(hashes)} hashes cargados de Qdrant")
    return hashes


def ensure_collection(client: QdrantClient, name: str):
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Colección creada: {name}")
    else:
        logger.info(f"Colección existente: {name}")


def upsert_batch(client: QdrantClient, collection: str, productos: list[Producto], openai: OpenAI):
    texts = [p.texto_para_embedding() for p in productos]
    resp = openai.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    vectors = [e.embedding for e in resp.data]
    points = [
        PointStruct(id=p.uid(), vector=v, payload=p.payload())
        for p, v in zip(productos, vectors)
    ]
    result = client.upsert(collection_name=collection, points=points)
    if result.status != UpdateStatus.COMPLETED:
        logger.warning(f"Upsert status inesperado: {result.status}")

# ============================================================
# Pipeline con Playwright stealth
# ============================================================

BROWSER_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
]


async def scrape_urls_with_playwright(
    urls: list[str],
    parser_fn,
    collection_name: str,
    qdrant: QdrantClient,
    openai_client: OpenAI,
):
    existing_hashes = fetch_existing_hashes(qdrant, collection_name)
    ok = skipped = err = 0
    batch: list[Producto] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=BROWSER_ARGS)
        context = await browser.new_context(
            locale="es-ES",
            timezone_id="Atlantic/Canary",
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        )

        for i, url in enumerate(urls):
            page: Page = await context.new_page()
            try:
                await stealth_async(page)
                await page.goto(url, timeout=PAGE_TIMEOUT, wait_until="domcontentloaded")
                html = await page.content()
                producto = parser_fn(html, url)

                if producto:
                    if existing_hashes.get(producto.uid()) == producto.content_hash():
                        skipped += 1
                    else:
                        batch.append(producto)
                        ok += 1
                else:
                    err += 1
            except Exception as e:
                logger.warning(f"Error scraping {url}: {e}")
                err += 1
            finally:
                await page.close()

            if len(batch) >= BATCH_SIZE:
                upsert_batch(qdrant, collection_name, batch, openai_client)
                logger.info(f"[{collection_name}] Upserted {ok} | Skipped {skipped} | Err {err} | Total {i+1}/{len(urls)}")
                batch = []

            await asyncio.sleep(SCRAPE_DELAY)

        await browser.close()

    if batch:
        upsert_batch(qdrant, collection_name, batch, openai_client)

    logger.info(f"[{collection_name}] COMPLETO — {ok} actualizados, {skipped} sin cambios, {err} errores")


async def run_pipeline_async(tienda: str):
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)

    if tienda in ("ventana_natural", "all"):
        ensure_collection(qdrant, "ventana_natural_productos")
        urls = get_product_urls_ventana_natural()
        logger.info(f"Procesando {len(urls)} URLs para ventana_natural_productos")
        await scrape_urls_with_playwright(urls, parse_ventana_natural, "ventana_natural_productos", qdrant, openai_client)

    if tienda in ("laurisilva", "all"):
        logger.info("Iniciando scraping laurisilva por categorías (sin Playwright)")
        scrape_laurisilva_categorias(qdrant, openai_client)


def run_pipeline(tienda: str):
    asyncio.run(run_pipeline_async(tienda))


if __name__ == "__main__":
    import sys
    tienda = sys.argv[1] if len(sys.argv) > 1 else "all"
    run_pipeline(tienda)
