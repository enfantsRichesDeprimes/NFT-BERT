/**
 *Submitted for verification at Etherscan.io on 2021-12-15
*/

pragma solidity ^0.8.10;
// SPDX-License-Identifier: MIT
abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");
        return c;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        return sub(a, b, "SafeMath: subtraction overflow");
    }

    function sub(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        require(b <= a, errorMessage);
        uint256 c = a - b;
        return c;
    }

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b, "SafeMath: multiplication overflow");
        return c;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        return div(a, b, "SafeMath: division by zero");
    }

    function div(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        require(b > 0, errorMessage);
        uint256 c = a / b;
        return c;
    }
}

contract Ownable is Context {
    address private _owner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        address msgSender = _msgSender();
        _owner = msgSender;
        emit OwnershipTransferred(address(0), msgSender);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(_owner == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        _setOwner(address(0));
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _setOwner(newOwner);
    }

    function _setOwner(address newOwner) private {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

interface IUniswapV2Factory {
    event PairCreated(address indexed token0, address indexed token1, address pair, uint);

    function feeTo() external view returns (address);
    function feeToSetter() external view returns (address);

    function getPair(address tokenA, address tokenB) external view returns (address pair);
    function allPairs(uint) external view returns (address pair);
    function allPairsLength() external view returns (uint);

    function createPair(address tokenA, address tokenB) external returns (address pair);

    function setFeeTo(address) external;
    function setFeeToSetter(address) external;
}

interface IUniswapV2Router02 {
    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external;
    function factory() external pure returns (address);
    function WETH() external pure returns (address);
    function addLiquidityETH(
        address token,
        uint256 amountTokenDesired,
        uint256 amountTokenMin,
        uint256 amountETHMin,
        address to,
        uint256 deadline
    ) external payable returns (uint256 amountToken, uint256 amountETH, uint256 liquidity);
}

contract NFTMusicStream is Context, IERC20, Ownable {
    using SafeMath for uint256;

    string private constant _name = "NFTMUSIC.STREAM";
    string private constant _symbol = "STREAMER";
    uint8 private constant _decimals = 9;

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) private _isExcludedFromFee;

    address public immutable deadAddress = address(0);

    uint256 private constant _tTotal = 1000000000000 * 1e9; //
    uint256 private _totalFee = 10;
    uint256 private _storedTotalFee = _totalFee;

    // For payout calculations
    uint256 public _payoutAdmin = 20;
    uint256 public _payoutMarketing = 40;
    uint256 public _payoutAppDev = 40;

    address payable private _adminAddress;
    address payable private _marketingAddress;
    address payable private _appDevAddress;
    mapping(address => bool) private _isAdmin;

    IUniswapV2Router02 private uniswapV2Router;
    address private uniswapV2Pair;
    bool private tradingOpen = false;
    bool private inSwap = false;
    bool private swapEnabled = false;
    bool private supportLiquidity = false;

    event SwapAndLiquify(uint256 tokensSwapped, uint256 ethReceived, uint256 tokensIntoLiquidity);
    event SwapTokensForETH(uint256 amountIn, address[] path);
    
    modifier lockTheSwap {
        inSwap = true;
        _;
        inSwap = false;
    }

    constructor(address payable adminFunds, address payable marketingFunds, address payable appDevFunds) {

        IUniswapV2Router02 _uniswapV2Router = IUniswapV2Router02(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        uniswapV2Pair = IUniswapV2Factory(_uniswapV2Router.factory()).createPair(address(this), _uniswapV2Router.WETH());
        uniswapV2Router = _uniswapV2Router;

        _adminAddress = adminFunds;
        _marketingAddress = marketingFunds;
        _appDevAddress = appDevFunds;
        
        _balances[_msgSender()] = _tTotal;
        _isExcludedFromFee[owner()] = true;
        _isAdmin[owner()] = true;
        _isExcludedFromFee[address(this)] = true;
        _isAdmin[address(this)] = true;
        _isExcludedFromFee[_adminAddress] = true;
        _isAdmin[_adminAddress] = true;
        _isExcludedFromFee[_marketingAddress] = true;
        _isAdmin[_marketingAddress] = true;
        _isExcludedFromFee[_appDevAddress] = true;
        _isAdmin[_appDevAddress] = true;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function name() public pure returns (string memory) {
        return _name;
    }

    function symbol() public pure returns (string memory) {
        return _symbol;
    }

    function decimals() public pure returns (uint8) {
        return _decimals;
    }

    function totalSupply() public pure override returns (uint256) {
        return _tTotal;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(sender,_msgSender(),_allowances[sender][_msgSender()].sub(amount,"ERC20: transfer amount exceeds allowance"));
        return true;
    }
    
    function removeAllFee() private {
        if (_totalFee == 0) return;
        _totalFee = 0;
    }

    function restoreAllFee() private {
        _totalFee = _storedTotalFee;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address from, address to, uint256 amount) private {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        require(amount > 0, "Transfer amount must be greater than zero");

        bool takeFee;

        if (!_isAdmin[from] && !_isAdmin[from]) {
            require(tradingOpen);
            takeFee = true;

            uint256 contractTokenBalance = balanceOf(address(this));
            if (!inSwap && from != uniswapV2Pair && swapEnabled) {

                if (supportLiquidity) {
                    uint256 liquidityPart = contractTokenBalance.div(2);
                    swapTokensForEth(liquidityPart);
                    uint256 newContractBalance = balanceOf(address(this));
                    swapAndLiquify(newContractBalance);
                } else {
                    swapTokensForEth(contractTokenBalance);
                }
                uint256 contractETHBalance = address(this).balance;
                if (contractETHBalance > 0) {
                    sendETHToWallets(address(this).balance);
                }
            }
        }

        if (_isExcludedFromFee[from] || _isExcludedFromFee[to]) {
            takeFee = false;
        }

        _tokenTransfer(from, to, amount, takeFee);
        restoreAllFee;
    }

    function swapTokensForEth(uint256 tokenAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = uniswapV2Router.WETH();
        _approve(address(this), address(uniswapV2Router), tokenAmount);
        uniswapV2Router.swapExactTokensForETHSupportingFeeOnTransferTokens(tokenAmount, 0, path, address(this), block.timestamp);
        emit SwapTokensForETH(tokenAmount, path);
    }

      function swapAndLiquify(uint256 contractTokenBalance) private lockTheSwap {
        uint256 half = contractTokenBalance.div(2);
        uint256 otherHalf = contractTokenBalance.sub(half);
        uint256 initialBalance = address(this).balance;
        swapTokensForEth(half);
        uint256 newBalance = address(this).balance.sub(initialBalance);
        addLiquidity(otherHalf, newBalance);
        emit SwapAndLiquify(half, newBalance, otherHalf);
    }
    

    function sendETHToWallets(uint256 totalETHbeforeSplit) private {
        if (_payoutAdmin != 0) {
            uint256 adminCut = totalETHbeforeSplit.mul(_payoutAdmin).div(100);
            _adminAddress.transfer(adminCut);
        }

        if (_payoutMarketing != 0) {
            uint256 marketingCut = totalETHbeforeSplit.mul(_payoutMarketing).div(100);
            _marketingAddress.transfer(marketingCut);
        }

        if (_payoutAppDev != 0) {
            uint256 appDevCut = totalETHbeforeSplit.mul(_payoutAppDev).div(100);
            _appDevAddress.transfer(appDevCut);
        }
    }
    
    function openTrading() public onlyOwner {
        tradingOpen = true;
    }

    function presaleFinished() external onlyOwner() {
        swapEnabled = true;
        supportLiquidity = true;
    }

    function addLiquidity(uint256 tokenAmount, uint256 ethAmount) private {
        _approve(address(this), address(uniswapV2Router), tokenAmount);
        uniswapV2Router.addLiquidityETH{value: ethAmount}(address(this), tokenAmount, 0, 0, address(this), block.timestamp);
    }

    function liquiditySupport(bool trueFalse) public onlyOwner {
        supportLiquidity = trueFalse;
    }

    function manualTokenSwap() external {
        require(_msgSender() == owner());
        uint256 contractBalance = balanceOf(address(this));
        swapTokensForEth(contractBalance);
    }

    function recoverEthFromContract() external {
        require(_msgSender() == owner());
        uint256 contractETHBalance = address(this).balance;
        sendETHToWallets(contractETHBalance);
    }

    function _tokenTransfer(address sender, address recipient, uint256 amount, bool takeFee) private {
        if (!takeFee) removeAllFee();
        _transferStandard(sender, recipient, amount);
        if (!takeFee) restoreAllFee();
    }

    function _transferStandard(address sender, address recipient, uint256 tAmount) private {
        (uint256 tTransferAmount, uint256 tTeam) = _getValues(tAmount);
        _balances[sender] = _balances[sender].sub(tAmount);
        _balances[recipient] = _balances[recipient].add(tTransferAmount);
        _takeTeam(tTeam);
        emit Transfer(sender, recipient, tTransferAmount);
    }

    function _takeTeam(uint256 tTeam) private {
        _balances[address(this)] = _balances[address(this)].add(tTeam);
    }

    receive() external payable {}

    function _getValues(uint256 tAmount) private view returns (uint256, uint256) {
        (uint256 tTransferAmount, uint256 tTeam) = _getTValues(tAmount, _totalFee);
        return (tTransferAmount, tTeam);
    }

    function _getTValues(uint256 tAmount, uint256 teamFee) private pure returns (uint256, uint256) {
        uint256 tTeam = tAmount.mul(teamFee).div(100);
        uint256 tTransferAmount = tAmount.sub(tTeam);
        return (tTransferAmount, tTeam);
    }

    function manualBurn (uint256 amount) external onlyOwner() {
        require(amount <= balanceOf(owner()), "Amount exceeds available tokens balance");
        _tokenTransfer(msg.sender, deadAddress, amount, false);
    }

    function setRouterAddress(address newRouter) public onlyOwner() {
        IUniswapV2Router02 _newUniRouter = IUniswapV2Router02(newRouter);
        uniswapV2Pair = IUniswapV2Factory(_newUniRouter.factory()).createPair(address(this), _newUniRouter.WETH());
        uniswapV2Router = _newUniRouter;
    }

    function setAddressAdmin(address payable newAdminAddress) external onlyOwner() {
        _isExcludedFromFee[_adminAddress] = false;
        _isAdmin[_adminAddress] = false;
        _adminAddress = newAdminAddress;
        _isExcludedFromFee[newAdminAddress] = true;
        _isAdmin[newAdminAddress] = true;
    }

    function setAddressMarketing(address payable newMarketingAddress) external onlyOwner() {
        _isExcludedFromFee[_marketingAddress] = false;
        _isAdmin[_marketingAddress] = false;
        _marketingAddress = newMarketingAddress;
        _isExcludedFromFee[newMarketingAddress] = true;
        _isAdmin[newMarketingAddress] = true;
    }

    function setAddressAppDev(address payable newAppDevAddress) external onlyOwner() {
        _isExcludedFromFee[_appDevAddress] = false;
        _isAdmin[_appDevAddress] = false;
        _appDevAddress = newAppDevAddress;
        _isExcludedFromFee[newAppDevAddress] = true;
        _isAdmin[newAppDevAddress] = true;
    }

    function setPayouts(uint256 newAdminPayout, uint256 newMarketingPayout, uint256 newAppDevPayout) external onlyOwner {
        require(newAdminPayout + newMarketingPayout + newAppDevPayout == 100, "Values must equal 100");
        _payoutAdmin = newAdminPayout;
        _payoutMarketing = newMarketingPayout;
        _payoutAppDev = newAppDevPayout;
    }
    
    function setFee(uint newFee) external onlyOwner {
        require(newFee <= 10, "Fee must be less than 10");
        _totalFee = newFee;
        _storedTotalFee = newFee;
    }

    function setIsAdmin(address payable newIsAdminAddress) external onlyOwner () {
      _isExcludedFromFee[newIsAdminAddress] = true;
      _isAdmin[newIsAdminAddress] = true;
    }

    function removeIsAdmin(address payable oldIsAdminAddress) external onlyOwner () {
      _isExcludedFromFee[oldIsAdminAddress] = false;
      _isAdmin[oldIsAdminAddress] = false;
    }
}