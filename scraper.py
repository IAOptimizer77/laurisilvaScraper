import os
import time
import logging
import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, UpdateStatus

# ============================================================
# Configuración
# ============================================================

QDRANT_URL        = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL   = "text-embedding-3-small"
EMBEDDING_DIM     = 1536
BATCH_SIZE        = 50
SCRAPE_DELAY      = float(os.getenv("SCRAPE_DELAY", "1.5"))  # segundos entre requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("scraper")

# ============================================================
# Modelos
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

    def id(self) -> str:
        return hashlib.md5(self.url.encode()).hexdigest()

    def payload(self) -> dict:
        return {
            "NombreProducto": self.nombre,
            "Precio":         self.precio,
            "Marca":          self.marca,
            "InformacionProducto": self.descripcion,
            "Categoria":      self.categoria,
            "Formato":        self.formato,
            "SKU":            self.sku,
            "URL":            self.url,
            "Tienda":         self.tienda,
        }

# ============================================================
# Sitemap parsers
# ============================================================

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

def get_urls_from_sitemap(sitemap_url: str, session: requests.Session) -> list[str]:
    try:
        r = session.get(sitemap_url, timeout=15)
        r.raise_for_status()
        content = r.content
        if content[:1] != b"<":
            logger.error(f"Sitemap {sitemap_url} no devolvió XML — posible bloqueo. Primeros 200 bytes: {content[:200]}")
            return []
        root = ET.fromstring(content)
        urls = [loc.text.strip() for loc in root.findall(".//sm:loc", SITEMAP_NS) if loc.text]
        logger.info(f"Sitemap {sitemap_url} → {len(urls)} URLs")
        return urls
    except Exception as e:
        logger.error(f"Error leyendo sitemap {sitemap_url}: {e}")
        return []


def get_product_urls_ventana_natural(sitemap_session: requests.Session) -> list[str]:
    """PrestaShop: filtra URLs de producto por patrón /categoria/id-nombre.html"""
    import re
    all_urls = get_urls_from_sitemap("https://laventananatural.com/1_es_0_sitemap.xml", sitemap_session)
    product_pattern = re.compile(r"laventananatural\.com/[^/]+/\d+-[^/]+\.html$")
    products = [u for u in all_urls if product_pattern.search(u)]
    logger.info(f"La Ventana Natural — {len(products)} productos encontrados")
    return products


def get_product_urls_laurisilva(sitemap_session: requests.Session) -> list[str]:
    """Laurisilvabio: 4 sitemaps paginados de productos"""
    sitemaps = [
        "https://www.laurisilvabio.com/product_sitemap_0.xml",
        "https://www.laurisilvabio.com/product_sitemap_1000.xml",
        "https://www.laurisilvabio.com/product_sitemap_2000.xml",
        "https://www.laurisilvabio.com/product_sitemap_3000.xml",
    ]
    all_urls = []
    for sm in sitemaps:
        all_urls.extend(get_urls_from_sitemap(sm, sitemap_session))
    logger.info(f"LauriSilvaBio — {len(all_urls)} productos encontrados")
    return all_urls

# ============================================================
# Scrapers por tienda
# ============================================================

def scrape_ventana_natural(url: str, session: requests.Session) -> Optional[Producto]:
    """Scraper para laventananatural.com (PrestaShop)"""
    try:
        r = session.get(url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        nombre = (
            soup.select_one("h1.product_name, h1.page-heading, h1[itemprop='name']") or
            soup.select_one("h1")
        )
        nombre = nombre.get_text(strip=True) if nombre else ""

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

        cat_breadcrumbs = soup.select("nav[aria-label='breadcrumb'] li, ol.breadcrumb li, .breadcrumb li")
        categoria = " > ".join(
            b.get_text(strip=True) for b in cat_breadcrumbs[1:-1]
        ) if len(cat_breadcrumbs) > 2 else ""

        sku_el = soup.select_one("[itemprop='sku'], .product-reference span")
        sku = sku_el.get_text(strip=True) if sku_el else url.split("-")[-1].replace(".html", "")

        if not nombre:
            return None

        return Producto(
            nombre=nombre, precio=precio, marca=marca,
            descripcion=descripcion, categoria=categoria,
            sku=sku, url=url, tienda="La Ventana Natural"
        )
    except Exception as e:
        logger.warning(f"Error scraping {url}: {e}")
        return None


def scrape_laurisilva(url: str, session: requests.Session) -> Optional[Producto]:
    """
    Scraper para laurisilvabio.com.
    Adaptar los selectores CSS según la plataforma real de la web.
    Punto de partida genérico con JSON-LD + fallback HTML.
    """
    import json as _json
    try:
        r = session.get(url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        # Intentar JSON-LD primero (WooCommerce/Shopify lo incluyen)
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = _json.loads(script.string or "")
                if isinstance(data, list):
                    data = next((d for d in data if d.get("@type") == "Product"), None)
                if data and data.get("@type") == "Product":
                    offers = data.get("offers", {})
                    if isinstance(offers, list): offers = offers[0] if offers else {}
                    precio = str(offers.get("price", "")) + " " + offers.get("priceCurrency", "")
                    marca_data = data.get("brand", {})
                    marca = marca_data.get("name", "") if isinstance(marca_data, dict) else str(marca_data)
                    return Producto(
                        nombre=data.get("name", ""),
                        precio=precio.strip(),
                        marca=marca,
                        descripcion=BeautifulSoup(data.get("description", ""), "lxml").get_text()[:1000],
                        categoria=data.get("category", ""),
                        sku=str(data.get("sku", "")),
                        url=url,
                        tienda="LauriSilvaBio",
                    )
            except Exception:
                continue

        # Fallback HTML genérico — ajustar selectores si no funciona
        nombre = soup.select_one("h1")
        precio_el = soup.select_one(".price, .product-price, [itemprop='price']")
        marca_el = soup.select_one(".brand, [itemprop='brand'], .manufacturer")
        desc_el = soup.select_one(".description, [itemprop='description'], .product-description")

        nombre_text = nombre.get_text(strip=True) if nombre else ""
        if not nombre_text:
            return None

        return Producto(
            nombre=nombre_text,
            precio=precio_el.get_text(strip=True) if precio_el else "",
            marca=marca_el.get_text(strip=True) if marca_el else "",
            descripcion=desc_el.get_text(separator=" ", strip=True)[:1000] if desc_el else "",
            categoria="",
            sku="",
            url=url,
            tienda="LauriSilvaBio",
        )
    except Exception as e:
        logger.warning(f"Error scraping {url}: {e}")
        return None

# ============================================================
# Qdrant — setup y upsert
# ============================================================

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
        PointStruct(
            id=int(p.id(), 16) % (2**63),
            vector=v,
            payload=p.payload(),
        )
        for p, v in zip(productos, vectors)
    ]
    result = client.upsert(collection_name=collection, points=points)
    if result.status != UpdateStatus.COMPLETED:
        logger.warning(f"Upsert status: {result.status}")

# ============================================================
# Pipeline principal
# ============================================================

def run_pipeline(tienda: str):
    """
    tienda: "ventana_natural" | "laurisilva" | "all"
    """
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)

    # Sesión para sitemaps XML — headers mínimos, sin fingerprinting de navegador
    sitemap_session = requests.Session()
    sitemap_session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
        "Accept": "application/xml,text/xml,*/*",
        "Accept-Language": "es-ES,es;q=0.9",
    })

    # Sesión para páginas de producto — simula navegador Chrome completo
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    })

    tasks = []
    if tienda in ("ventana_natural", "all"):
        tasks.append(("ventana_natural_productos", get_product_urls_ventana_natural, scrape_ventana_natural))
    if tienda in ("laurisilva", "all"):
        tasks.append(("laurisilva_productos", get_product_urls_laurisilva, scrape_laurisilva))

    # Warm-up: visitar home de cada tienda antes de scrapear productos
    warmup_urls = {
        "ventana_natural_productos": "https://laventananatural.com/",
        "laurisilva_productos":      "https://www.laurisilvabio.com/",
    }

    for collection_name, get_urls_fn, scrape_fn in tasks:
        try:
            warmup = warmup_urls.get(collection_name, "")
            if warmup:
                session.get(warmup, timeout=10)
                time.sleep(2)
        except Exception:
            pass

        ensure_collection(qdrant, collection_name)
        urls = get_urls_fn(sitemap_session)
        logger.info(f"Procesando {len(urls)} URLs para {collection_name}")

        batch: list[Producto] = []
        ok = err = 0

        for i, url in enumerate(urls):
            producto = scrape_fn(url, session)
            if producto:
                batch.append(producto)
                ok += 1
            else:
                err += 1

            if len(batch) >= BATCH_SIZE:
                upsert_batch(qdrant, collection_name, batch, openai_client)
                logger.info(f"[{collection_name}] Upserted {ok} / {i+1} procesados, {err} errores")
                batch = []

            time.sleep(SCRAPE_DELAY)

        if batch:
            upsert_batch(qdrant, collection_name, batch, openai_client)

        logger.info(f"[{collection_name}] COMPLETO — {ok} productos indexados, {err} errores")


if __name__ == "__main__":
    import sys
    tienda = sys.argv[1] if len(sys.argv) > 1 else "all"
    run_pipeline(tienda)
