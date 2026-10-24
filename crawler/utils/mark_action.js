// find element by xpath
function getElementByXPath(xpath) {
    const result = document.evaluate(
        xpath,
        document,
        null,
        XPathResult.FIRST_ORDERED_NODE_TYPE,
        null
    );
    return result.singleNodeValue;
}


// highlight the element
function highlightElement(xpath, actionLabel, elementText = "") {
    const element = getElementByXPath(xpath);
    if (!element) {
        console.error('Element not found:', xpath);
        return false;
    }
    
    // 滚动元素到视口中心
    // element.scrollIntoView({ 
    //     behavior: 'instant', 
    //     block: 'center',
    //     inline: 'center'
    // });
    
    // get element position
    const rect = element.getBoundingClientRect();
    
    // create highlight box
    const highlightBox = document.createElement('div');
    highlightBox.id = 'playwright-highlight-box';
    highlightBox.style.cssText = `
        position: fixed;
        left: ${rect.left}px;
        top: ${rect.top}px;
        width: ${rect.width}px;
        height: ${rect.height}px;
        border: 3px solid #ff4444;
        background: rgba(255, 68, 68, 0.1);
        pointer-events: none;
        z-index: 999999;
        box-shadow: 0 0 10px rgba(255, 68, 68, 0.5);
        border-radius: 4px;
    `;
    document.body.appendChild(highlightBox);
    
    // // create action label
    // const label = document.createElement('div');
    // label.id = 'playwright-action-label';

    // // build label content
    // let labelContent = `<strong style="font-size: 14px;">${actionLabel}</strong>`;
    // if (elementText && elementText.length > 0) {
    //     const displayText = elementText.length > 30 
    //         ? elementText.substring(0, 30) + '...' 
    //         : elementText;
    //     labelContent += `<br><small style="font-size: 12px; opacity: 0.9;">${displayText}</small>`;
    // }
    
    // label.innerHTML = labelContent;
    // label.style.cssText = `
    //     position: fixed;
    //     left: ${rect.left}px;
    //     top: ${rect.top - 50}px;
    //     background: linear-gradient(135deg, #ff4444 0%, #ff6666 100%);
    //     color: white;
    //     padding: 8px 14px;
    //     border-radius: 6px;
    //     font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    //     font-size: 14px;
    //     pointer-events: none;
    //     z-index: 1000000;
    //     box-shadow: 0 3px 12px rgba(0, 0, 0, 0.4);
    //     white-space: nowrap;
    //     line-height: 1.4;
    // `;
    
    // // if label goes above viewport, place it below the element
    // if (rect.top - 50 < 0) {
    //     label.style.top = `${rect.bottom + 8}px`;
    // }
    
    // // if label goes beyond right viewport, right-align it
    // if (rect.left + 200 > window.innerWidth) {
    //     label.style.left = 'auto';
    //     label.style.right = `${window.innerWidth - rect.right}px`;
    // }
    
    // document.body.appendChild(label);
    
    return true;
}


// remove highlight
function removeHighlight() {
    const highlightBox = document.getElementById('playwright-highlight-box');
    // const actionLabel = document.getElementById('playwright-action-label');
    
    if (highlightBox) {
        highlightBox.remove();
    }
    
    // if (actionLabel) {
    //     actionLabel.remove();
    // }
    
    return true;
}


// ====================================================================================
// 扩展函数：高亮多个元素（可选）
// ====================================================================================
function highlightMultipleElements(elements) {
    /**
     * 高亮多个元素
     * @param {Array} elements - 元素数组，每个元素包含 {xpath, actionLabel, elementText}
     */
    elements.forEach((elemData, index) => {
        const element = getElementByXPath(elemData.xpath);
        if (!element) return;
        
        const rect = element.getBoundingClientRect();
        
        // 创建高亮框
        const highlightBox = document.createElement('div');
        highlightBox.className = 'playwright-multi-highlight';
        highlightBox.style.cssText = `
            position: fixed;
            left: ${rect.left}px;
            top: ${rect.top}px;
            width: ${rect.width}px;
            height: ${rect.height}px;
            border: 2px solid #4444ff;
            background: rgba(68, 68, 255, 0.05);
            pointer-events: none;
            z-index: 999998;
        `;
        document.body.appendChild(highlightBox);
        
        // 创建序号标签
        const badge = document.createElement('div');
        badge.className = 'playwright-multi-badge';
        badge.textContent = index + 1;
        badge.style.cssText = `
            position: fixed;
            left: ${rect.left - 5}px;
            top: ${rect.top - 5}px;
            width: 24px;
            height: 24px;
            background: #4444ff;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 12px;
            pointer-events: none;
            z-index: 999999;
        `;
        document.body.appendChild(badge);
    });
    
    return true;
}


// ====================================================================================
// 清理函数：移除所有多个高亮
// ====================================================================================
function removeMultipleHighlights() {
    const highlights = document.querySelectorAll('.playwright-multi-highlight');
    const badges = document.querySelectorAll('.playwright-multi-badge');
    
    highlights.forEach(el => el.remove());
    badges.forEach(el => el.remove());
    
    return true;
}