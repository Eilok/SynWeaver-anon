// ====================================================================================
// 步骤一：重写 addEventListener 来捕获事件 (黑科技)
// ====================================================================================
if (!window.eventListenersAttached) {
    const originalAddEventListener = EventTarget.prototype.addEventListener;
    EventTarget.prototype.addEventListener = function (type, listener, options) {
        if (!this._listeners) {
            this._listeners = [];
        }
        if (type === 'click') {
            this._listeners.push({ type, listener, options });
        }
        originalAddEventListener.call(this, type, listener, options);
    };
    window.eventListenersAttached = true;
}

// ====================================================================================
// 核心函数：markPage (改进版)
// ====================================================================================
const customCSS = `
::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-track {
    background: #27272a;
}
::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 0.375rem;
}
::-webkit-scrollbar-thumb:hover {
    background: #555;
}
`;

const styleTag = document.createElement("style");
styleTag.textContent = customCSS;
document.head.append(styleTag);

let labels = [];

function unmarkPage() {
    for (const label of labels) {
        if (label.parentElement) {
            label.parentElement.removeChild(label);
        }
    }
    labels = [];
}

// ========================================================================
// XPath 生成函数
// ========================================================================
function getXPath(element) {
    if (element === document.documentElement) {
        return '/html';
    }
    
    if (element === document.body) {
        return '/html/body';
    }
    
    if (element.parentNode === null) {
        return '/' + element.tagName.toLowerCase();
    }

    let ix = 0;
    const siblings = element.parentNode.childNodes;
    for (let i = 0; i < siblings.length; i++) {
        const sibling = siblings[i];
        if (sibling === element) {
            return `${getXPath(element.parentNode)}/${element.tagName.toLowerCase()}[${ix + 1}]`;
        }
        if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
            ix++;
        }
    }
    return 'XPath generation failed';
}

function markPage() {
    unmarkPage();

    function queryAllElements(root) {
        const elements = Array.from(root.querySelectorAll('*'));
        const allElements = [...elements];
        elements.forEach(el => {
            if (el.shadowRoot) {
                allElements.push(...queryAllElements(el.shadowRoot));
            }
        });
        return allElements;
    }

    var vw = Math.max(
        document.documentElement.clientWidth || 0,
        window.innerWidth || 0
    );
    var vh = Math.max(
        document.documentElement.clientHeight || 0,
        window.innerHeight || 0
    );

    var items = queryAllElements(document)
        .map(function (element) {
            var textualContent = element.textContent.trim().replace(/\s{2,}/g, " ");
            var elementType = element.tagName.toLowerCase();
            var ariaLabel = element.getAttribute("aria-label") || "";
            var role = element.getAttribute("role") || "";
            var name = element.getAttribute("name") || "";

            // 🔑 关键过滤：排除 SVG 基础图形元素（它们不可直接触发）
            const svgPrimitiveElements = [
                'use', 'path', 'circle', 'rect', 'line', 'polygon', 'polyline', 
                'ellipse', 'g', 'defs', 'symbol', 'clippath', 'mask', 'pattern',
                'lineargradient', 'radialgradient', 'stop', 'text', 'tspan', 'svg'
            ];
            if (svgPrimitiveElements.includes(elementType)) {
                return null;  // 直接排除
            }

            // 判断是否为可交互元素
            const isClickableByTag = ["a", "button", "select", "details", "summary"].includes(elementType);
            const isClickableByInputType = elementType === "input" && element.type !== "hidden";
            const isClickableByJs = element.onclick != null || (element._listeners && element._listeners.some(l => l.type === 'click'));
            const isClickableByCss = window.getComputedStyle(element).cursor == "pointer";
            const isClickableByRole = ["button", "link", "tab", "menuitem", "checkbox", "radio", "switch", "treeitem"].includes(role);
            const isSpecialElement = ["textarea", "iframe", "video"].includes(elementType);

            let isLabelForInteractive = false;
            if (elementType === 'label') {
                const forId = element.getAttribute('for');
                if (forId) {
                    const inputElement = document.getElementById(forId);
                    if (inputElement) {
                        isLabelForInteractive = true;
                    }
                }
            }

            const bb = element.getBoundingClientRect();
            
            var center_x = bb.left + bb.width / 2;
            var center_y = bb.top + bb.height / 2;
            var elAtCenter = document.elementFromPoint(center_x, center_y);
            
            const isVisible = elAtCenter === element || element.contains(elAtCenter);
            
            if (!isVisible) {
                return null;
            }

            const rect = {
                left: Math.max(0, bb.left),
                top: Math.max(0, bb.top),
                right: Math.min(vw, bb.right),
                bottom: Math.min(vh, bb.bottom),
            };
            
            rect.width = rect.right - rect.left;
            rect.height = rect.bottom - rect.top;
            const area = rect.width * rect.height;

            return {
                element: element,
                include:
                    isClickableByTag ||
                    isClickableByInputType ||
                    isClickableByJs ||
                    isClickableByCss ||
                    isClickableByRole ||
                    isSpecialElement ||
                    isLabelForInteractive,
                area,
                rect: rect,
                text: textualContent,
                tag: elementType,
                ariaLabel: ariaLabel,
                name: name,
                xpath: getXPath(element),
            };
        })
        .filter((item) => item !== null && item.include && item.area >= 20);

    // 过滤嵌套元素（保留最外层）
    items = items.filter(
        (x) => !items.some((y) => x.element.contains(y.element) && !(x == y))
    );

    // 🔑 关键修复：先分配 page_id，再提取列表结构
    items.forEach(function (item, index) {
        item.page_id = index;
    });

    // 识别列表容器及其子元素
    const listContainers = _extractListStructures(items);

    // 🆕 识别表单结构
    const formStructures = _extractFormStructures(items);

    function getRandomColor() {
        var letters = "0123456789ABCDEF";
        var color = "#";
        for (var i = 0; i < 6; i++) {
            color += letters[Math.floor(Math.random() * 16)];
        }
        return color;
    }

    // 创建标注
    items.forEach(function (item) {
        const bbox = item.rect;
        const borderColor = getRandomColor();
        
        const newElement = document.createElement("div");
        newElement.style.outline = `2px dashed ${borderColor}`;
        newElement.style.position = "fixed";
        newElement.style.left = bbox.left + "px";
        newElement.style.top = bbox.top + "px";
        newElement.style.width = bbox.width + "px";
        newElement.style.height = bbox.height + "px";
        newElement.style.pointerEvents = "none";
        newElement.style.boxSizing = "border-box";
        newElement.style.zIndex = 2147483647;

        const label = document.createElement("span");
        label.textContent = item.page_id;
        label.style.position = "absolute";
        label.style.top = "-19px";
        label.style.left = "0px";
        label.style.background = borderColor;
        label.style.color = "white";
        label.style.padding = "2px 4px";
        label.style.fontSize = "12px";
        label.style.borderRadius = "2px";
        newElement.appendChild(label);

        document.body.appendChild(newElement);
        labels.push(newElement);
    });

    // 构建返回结果
    const result = items.map((item) => {
        const rect = item.rect;
        
        return {
            page_id: item.page_id,
            tag: item.tag,
            text: item.text,
            ariaLabel: item.ariaLabel,
            name: item.name,
            xpath: item.xpath,
            // 返回完整的 bounding box 信息
            boundingBox: {
                left: rect.left,
                top: rect.top,
                right: rect.right,
                bottom: rect.bottom,
                width: rect.width,
                height: rect.height
            }
        };
    });

    return {
        elements: result,
        listStructures: listContainers,
        formStructures: formStructures
    };
}

// 🆕 提取列表结构的函数（改进版：基于 DOM 结构，返回列表内所有可交互元素）
function _extractListStructures(items) {
    const result = [];
    
    // 🔑 核心改进：直接从 DOM 查找列表容器
    const listContainerSelectors = [
        'ul',
        'ol',
        'table tbody',  // 表格优先匹配 tbody
        'table',
        '[role="list"]',
        '[role="listbox"]',
        '[role="menu"]',
    ];
    
    // 收集所有列表容器（去重）
    const allContainers = new Set();
    listContainerSelectors.forEach(selector => {
        document.querySelectorAll(selector).forEach(el => allContainers.add(el));
    });
    
    allContainers.forEach(container => {
        const containerTag = container.tagName.toLowerCase();
        
        // 确定列表项选择器
        let listItemSelector;
        if (containerTag === 'ul' || containerTag === 'ol' || container.getAttribute('role') === 'list') {
            listItemSelector = ':scope > li';
        } else if (containerTag === 'tbody' || containerTag === 'table') {
            listItemSelector = ':scope > tr';
        } else if (container.getAttribute('role') === 'listbox' || container.getAttribute('role') === 'menu') {
            listItemSelector = ':scope > [role="option"], :scope > [role="menuitem"], :scope > li';
        } else {
            listItemSelector = ':scope > li, :scope > div, :scope > article';
        }
        
        // 获取直接子列表项
        const listItems = Array.from(container.querySelectorAll(listItemSelector));
        
        // 跳过项目太少的列表（< 3 个）
        if (listItems.length < 3) return;
        
        // 🔑 核心修改：收集列表内所有可交互元素，而不是每个 li 只保留一个
        const allContainedItems = [];
        listItems.forEach((listItem, listItemIndex) => {
            // 找到该列表项内部包含的所有可交互元素
            const containedItems = items.filter(item => listItem.contains(item.element));
            
            containedItems.forEach(item => {
                allContainedItems.push({
                    page_id: item.page_id,
                    xpath: item.xpath,  // 🔑 使用可交互元素自己的 xpath
                    tag: item.tag,
                    text: item.text,
                    ariaLabel: item.ariaLabel,
                    name: item.name,
                    // 🆕 新增：该元素所属的列表项索引（0-based）
                    listItemIndex: listItemIndex,
                    // 🆕 新增：列表项的 xpath
                    listItemXPath: getXPath(listItem),
                });
            });
        });
        
        // 只保留有可交互元素 >= 3 个的列表
        if (allContainedItems.length >= 3) {
            result.push({
                containerXPath: getXPath(container),
                containerTag: containerTag,
                listItemCount: listItems.length,  // 🆕 列表项数量
                items: allContainedItems,  // 🔑 所有可交互元素
            });
        }
    });
    
    // 🔑 去重：如果一个容器是另一个容器的子集，保留更外层的
    return _deduplicateListContainers(result);
}

// 🆕 去重：移除被包含的子列表
function _deduplicateListContainers(containers) {
    // 按 items 数量降序排序（优先保留大列表）
    containers.sort((a, b) => b.items.length - a.items.length);
    
    const result = [];
    const usedPageIds = new Set();
    
    for (const container of containers) {
        // 检查该容器的 items 是否大部分已被使用
        const containerPageIds = container.items.map(item => item.page_id);
        const overlapCount = containerPageIds.filter(id => usedPageIds.has(id)).length;
        const overlapRatio = containerPageIds.length > 0 ? overlapCount / containerPageIds.length : 0;
        
        // 如果重叠率 < 50%，保留该容器
        if (overlapRatio < 0.5) {
            result.push(container);
            containerPageIds.forEach(id => usedPageIds.add(id));
        }
    }
    
    return result;
}


// ========================================================================
// 🆕 表单结构识别函数
// ========================================================================

/**
 * 提取表单结构
 * 识别页面上的 <form> 元素，收集其中的输入字段和提交按钮
 * 
 * @param {Array} items - 所有可交互元素列表
 * @returns {Array} 表单结构数组
 */
function _extractFormStructures(items) {
    const formContainers = {};
    
    items.forEach((item) => {
        const formAncestor = _findFormAncestorElement(item.element);
        
        if (formAncestor) {
            const formXPath = getXPath(formAncestor);
            
            if (!formContainers[formXPath]) {
                formContainers[formXPath] = {
                    containerXPath: formXPath,
                    inputFields: [],
                    submitButton: null
                };
            }
            
            // 分类：输入字段 vs 提交按钮
            if (_isInputField(item.element)) {
                formContainers[formXPath].inputFields.push({
                    page_id: item.page_id,
                    xpath: item.xpath,
                    tag: item.tag,
                    text: item.text,
                    ariaLabel: item.ariaLabel,
                    name: item.name,
                });
            } else if (_isSubmitButton(item.element)) {
                // 只保留第一个 submit 按钮
                if (!formContainers[formXPath].submitButton) {
                    // 🆕 获取真正的 submit button 信息（可能是当前元素的祖先）
                    const actualButton = _findActualSubmitButton(item.element);
                    formContainers[formXPath].submitButton = {
                        page_id: item.page_id,
                        xpath: item.xpath,  // 保持原始 xpath，因为点击子元素也能触发
                        tag: actualButton ? actualButton.tagName.toLowerCase() : item.tag,
                        text: item.text,
                        ariaLabel: actualButton ? (actualButton.getAttribute('aria-label') || item.ariaLabel) : item.ariaLabel,
                        name: actualButton ? (actualButton.getAttribute('name') || item.name) : item.name,
                    };
                }
            }
        }
    });
    
    // 只返回有输入字段且有提交按钮的表单
    return Object.values(formContainers).filter(
        form => form.inputFields.length > 0 && form.submitButton !== null
    );
}

/**
 * 查找表单祖先元素
 * 
 * @param {Element} element - 目标元素
 * @returns {Element|null} 表单元素或 null
 */
function _findFormAncestorElement(element) {
    let current = element.parentElement;
    
    while (current && current !== document.body) {
        if (current.tagName.toLowerCase() === 'form') {
            return current;
        }
        current = current.parentElement;
    }
    
    return null;
}

/**
 * 判断元素是否为输入字段
 * 包括：input（非 submit/button/hidden）、textarea、select
 * 🔧 修复：支持检测被包裹在 label 等容器内的输入字段
 * 
 * @param {Element} element - 目标元素
 * @returns {boolean}
 */
function _isInputField(element) {
    // 首先检查元素本身
    if (_isInputFieldDirect(element)) {
        return true;
    }
    
    // 🆕 向上查找是否有输入字段祖先（在 form 内部）
    let current = element.parentElement;
    while (current && current.tagName.toLowerCase() !== 'form' && current !== document.body) {
        if (_isInputFieldDirect(current)) {
            return true;
        }
        current = current.parentElement;
    }
    
    return false;
}

/**
 * 直接判断元素本身是否为输入字段
 * @param {Element} element - 目标元素
 * @returns {boolean}
 */
function _isInputFieldDirect(element) {
    const tag = element.tagName.toLowerCase();
    
    if (tag === 'textarea' || tag === 'select') {
        return true;
    }
    
    if (tag === 'input') {
        const type = (element.type || 'text').toLowerCase();
        // 排除不可填写的 input 类型
        return !['submit', 'button', 'reset', 'hidden', 'image'].includes(type);
    }
    
    return false;
}

/**
 * 判断元素是否为提交按钮
 * 🔧 修复：支持检测 button 内部的子元素（如 span、svg 等）
 * 
 * @param {Element} element - 目标元素
 * @returns {boolean}
 */
function _isSubmitButton(element) {
    // 首先检查元素本身
    if (_isSubmitButtonDirect(element)) {
        return true;
    }
    
    // 🆕 向上查找是否有 submit button 祖先（在 form 内部）
    // 解决 <button><span>Submit</span></button> 情况下 span 被识别为可触发元素的问题
    let current = element.parentElement;
    while (current && current.tagName.toLowerCase() !== 'form' && current !== document.body) {
        if (_isSubmitButtonDirect(current)) {
            return true;
        }
        current = current.parentElement;
    }
    
    return false;
}

/**
 * 直接判断元素本身是否为提交按钮
 * @param {Element} element - 目标元素
 * @returns {boolean}
 */
function _isSubmitButtonDirect(element) {
    const tag = element.tagName.toLowerCase();
    
    // input type="submit"
    if (tag === 'input' && element.type === 'submit') {
        return true;
    }
    
    // button (默认 type 为 submit，除非显式设置为 button 或 reset)
    if (tag === 'button') {
        const type = (element.type || 'submit').toLowerCase();
        return type === 'submit';
    }
    
    return false;
}

/**
 * 🆕 查找真正的 submit button 元素
 * 当检测到的可触发元素是 button 的子元素（如 span、svg）时，
 * 向上查找真正的 button/input[type=submit] 元素
 * 
 * @param {Element} element - 目标元素
 * @returns {Element|null} 真正的 submit button 元素，或 null（如果元素本身就是）
 */
function _findActualSubmitButton(element) {
    // 如果元素本身就是 submit button，返回 null（表示不需要查找）
    if (_isSubmitButtonDirect(element)) {
        return null;
    }
    
    // 向上查找
    let current = element.parentElement;
    while (current && current.tagName.toLowerCase() !== 'form' && current !== document.body) {
        if (_isSubmitButtonDirect(current)) {
            return current;
        }
        current = current.parentElement;
    }
    
    return null;
}