import asyncio
import hashlib
import json as _json
import logging
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

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

    tasks = []
    if tienda in ("ventana_natural", "all"):
        tasks.append(("ventana_natural_productos", lambda: get_product_urls_ventana_natural(), parse_ventana_natural))
    if tienda in ("laurisilva", "all"):
        tasks.append(("laurisilva_productos", lambda: get_product_urls_laurisilva(qdrant), parse_laurisilva))

    for collection_name, get_urls_fn, parser_fn in tasks:
        ensure_collection(qdrant, collection_name)
        urls = get_urls_fn()
        logger.info(f"Procesando {len(urls)} URLs para {collection_name}")
        await scrape_urls_with_playwright(urls, parser_fn, collection_name, qdrant, openai_client)


def run_pipeline(tienda: str):
    asyncio.run(run_pipeline_async(tienda))


if __name__ == "__main__":
    import sys
    tienda = sys.argv[1] if len(sys.argv) > 1 else "all"
    run_pipeline(tienda)
